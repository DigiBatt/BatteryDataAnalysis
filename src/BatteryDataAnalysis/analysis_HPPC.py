import pandas as pd
import numpy as np
from .create_custom_functions import add_function
from scipy.interpolate import interp1d


def pulse_number_HPPC(df_input):
    """Pulse selection for HPPC data

    For every cycles, gives a pulse number for every HPPC pattern (containing a charge and discharge pulse followed by a relaxation time).
    Then it gives this number for the last point before the first pulse until the last point before the end of the relaxation
    Finally, it creates a nested Dataframe structuring the HPPC data for each pattern (called 'Pulse' to be consistent with GITT and ICI processing)

    Here is a schema of a selected pulse and its relevant points:

    .. image:: ../../../../_static/pulse_HPPC.png
        :width: 2000px
        :align: center
        :alt: Example of a pulse selected by the function for HPPC data

    .. image:: ../../../../_static/pulse_HPPC_zoom.png
        :width: 2000px
        :align: center
        :alt: Current pulses part of a pulse selected by the function for HPPC data

    The corresponding parameters that can be used in a custom function (see :func:`add_function`) are:
    *V0, V1, V2, V3, V4, V5, t0, t1, t2, t3, t4, t5, Icharge, Idischarge* that are calculated in :func:`calculate_relevant_points_HPPC`.

    Parameters
    ----------
    df_input : pandas.DataFrame
        DataFrame containing the HPPC data to process

    Returns
    -------
    dict
        Dict containing the nested DataFrames as values and their pulse number as keys
    """
    # A pulse starts the time just before the pulse and ends the time just before the next pulse at the end of the relaxation time
    df = df_input.copy()

    pulse_count = 0
    for cycle in df["Cycle"].unique():
        for state in df[df["Cycle"] == cycle]["State"].unique():
            df_state = df[(df["State"] == state) & (df["Cycle"] == cycle)].copy()

            # Count the number of positive and negative current pulses (including relaxation)
            df_state["raw_neg_pulse_count"] = (
                (df_state["normcurrent"] < 0).astype(int).diff().fillna(0).gt(0).cumsum().ffill().shift(-1)
            )
            df_state["raw_pos_pulse_count"] = (
                (df_state["normcurrent"] > 0).astype(int).diff().fillna(0).gt(0).cumsum().ffill().shift(-1)
            )
            df_state = df_state.dropna(subset=["raw_neg_pulse_count", "raw_pos_pulse_count"])

            # If charging the only negative current is the HPPC pulse
            if state == "C":
                df_state["discharge_pulse"] = (df_state["normcurrent"] < 0).astype(int)
                # df_state['charge_pulse'] = 0

                df_group = (
                    df_state[df_state["normcurrent"] < 0]
                    .groupby("raw_neg_pulse_count")["TestTime"]
                    .agg(lambda x: x.max() - x.min())
                )
                pulse_time = df_group.mean()

                # If a positive pulse is the same length as the negative pulse then it is the HPPC pulse (and not the relaxation)
                for pulse in df_state["raw_pos_pulse_count"].unique():
                    df_pulse = df_state[(df_state["raw_pos_pulse_count"] == pulse) & (df_state["normcurrent"] > 0)]
                    if df_pulse["TestTime"].max() - df_pulse["TestTime"].min() < pulse_time * 1.5:  # 1.5
                        df_state.loc[
                            (df_state["raw_pos_pulse_count"] == pulse) & (df_state["normcurrent"] > 0), "charge_pulse"
                        ] = 1
                        pulse_count += 1

                    df_state.loc[df_state["raw_pos_pulse_count"] == pulse, "pos_pulse_count"] = pulse_count

                df_state["pos_pulse_count"] = df_state["pos_pulse_count"].ffill()
                df_state["Pulse"] = np.maximum(df_state["raw_neg_pulse_count"], df_state["pos_pulse_count"])

            # If discharging the only positive current is the HPPC pulse
            elif state == "D":
                df_state["charge_pulse"] = (df_state["normcurrent"] > 0).astype(int)
                # df_state['discharge_pulse'] = 0

                df_group = (
                    df_state[df_state["normcurrent"] > 0]
                    .groupby("raw_neg_pulse_count")["TestTime"]
                    .agg(lambda x: x.max() - x.min())
                )
                pulse_time = df_group.mean()

                # If a negative pulse is the same length as the positive pulse then it is the HPPC pulse (and not the relaxation)
                for pulse in df_state["raw_neg_pulse_count"].unique():
                    df_pulse = df_state[(df_state["raw_neg_pulse_count"] == pulse) & (df_state["normcurrent"] < 0)]
                    if df_pulse["TestTime"].max() - df_pulse["TestTime"].min() < pulse_time * 1.5:  # 1.5
                        df_state.loc[
                            (df_state["raw_neg_pulse_count"] == pulse) & (df_state["normcurrent"] < 0),
                            "discharge_pulse",
                        ] = 1
                        pulse_count += 1

                    df_state.loc[df_state["raw_neg_pulse_count"] == pulse, "neg_pulse_count"] = pulse_count

                df_state["neg_pulse_count"] = df_state["neg_pulse_count"].ffill()
                df_state["Pulse"] = np.maximum(df_state["neg_pulse_count"], df_state["raw_pos_pulse_count"])

            df.loc[(df["State"] == state) & (df["Cycle"] == cycle), "Pulse"] = df_state["Pulse"].astype(int)
            df.loc[(df["State"] == state) & (df["Cycle"] == cycle), "charge_pulse"] = df_state["charge_pulse"]
            df.loc[(df["State"] == state) & (df["Cycle"] == cycle), "discharge_pulse"] = df_state["discharge_pulse"]

    df = df.dropna(subset=["Pulse"])
    df_nested = {int(pulse): df[df["Pulse"] == pulse] for pulse in df["Pulse"].unique()}

    return df_nested


def global_calculation_HPPC(df_nested, my_func_list):
    """Calculates the HPPC parameters

    Given a nested DataFrame containing the HPPC data, this function calculates the HPPC parameters for each pulse

    Parameters
    ----------
    df_nested : Dict
        Dict containing the nested DataFrames corresponding to each pulse

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the HPPC parameters structured by pulse number
    """
    df_total = pd.DataFrame()

    df = pd.concat(df_nested.values())
    Vmin = df["Voltage"].min()
    Vmax = df["Voltage"].max()

    for pulse, df_pulse in df_nested.items():
        df_charge = df_pulse[df_pulse["charge_pulse"] == 1]
        df_discharge = df_pulse[df_pulse["discharge_pulse"] == 1]

        if len(df_charge) >= 2 and len(df_discharge) >= 2 and pulse != 0:
            try:
                V0, V1, V2, V3, V4, V5, t0, t1, t2, t3, t4, t5, Idischarge, Icharge = calculate_relevant_points_HPPC(
                    df_pulse
                )

                R_discharge = abs((V1 - V0) / Idischarge)
                R_charge = abs((V4 - V3) / Icharge)
                P_discharge, P_charge = calculate_pulse_power_capability(Vmin, Vmax, V0, R_discharge, R_charge)

                df_coefficient = pd.DataFrame(
                    {
                        "Pulse": pulse,
                        "Cycle": df_pulse["Cycle"].iloc[0],
                        "State": df_pulse["State"].iloc[0],
                        "TestTime": df_pulse["TestTime"].iloc[0],
                        "SOC": df_pulse["SOC"].iloc[0],
                        "OCV": V0,
                        "R_charge": R_charge,
                        "R_discharge": R_discharge,
                        "P_charge": P_charge,
                        "P_discharge": P_discharge,
                    },
                    index=["Pulse"],
                )

                # Parameters that can be used in an external function
                kwargs = {
                    "V0": V0,
                    "V1": V1,
                    "V2": V2,
                    "V3": V3,
                    "V4": V4,
                    "V5": V5,
                    "t0": t0,
                    "t1": t1,
                    "t2": t2,
                    "t3": t3,
                    "t4": t4,
                    "t5": t5,
                    "Idischarge": Idischarge,
                    "Icharge": Icharge,
                    "R_charge": R_charge,
                    "R_discharge": R_discharge,
                    "Voltage": interp1d(df_pulse["TestTime"], df_pulse["Voltage"], kind="linear"),
                    "Current": interp1d(df_pulse["TestTime"], df_pulse["Current"], kind="linear"),
                    "Capacity": interp1d(df_pulse["TestTime"], df_pulse["Capacity"], kind="linear"),
                    "State": df_pulse["State"].iloc[0],
                }

                for func in my_func_list:
                    var_name = func.__name__
                    new_var = add_function(func, **kwargs)
                    if new_var is not None:
                        df_coefficient[var_name] = new_var

                df_total = pd.concat([df_total, df_coefficient])

            except Exception as e:
                print(f"Error processing pulse {pulse}: {e}")

    return df_total


def calculate_relevant_points_HPPC(df_pulse):
    """Calculates the relevant points represented in the :func:`pulse_number_GITT` documentation.

    These parameters can be used in a custom function to calculate new parameters (see :func:`add_function`).

    Parameters
    ----------
    df_pulse : pandas.DataFrame
        DataFrame containing the data corresponding to one pulse

    Returns
    -------
    tuple
        Tuple containing the relevant points : V0, V1, V2, V3, V4, V5, t0, t1, t2, t3, t4, t5, Icharge, Idischarge
    """
    df = df_pulse.copy()

    # Relevant points calculation
    # Discharge pulse
    df_discharge = df_pulse[df_pulse["discharge_pulse"] == 1]
    t1 = df_discharge["TestTime"].iloc[0]
    V1 = df_discharge["Voltage"].iloc[0]

    t2 = df_discharge["TestTime"].iloc[-1]
    V2 = df_discharge["Voltage"].iloc[-1]

    t0 = df[(df["TestTime"] < t1) & (df["normcurrent"] == 0)]["TestTime"].iloc[-1]
    V0 = df[df["TestTime"] == t0]["Voltage"].mean()
    Idischarge = abs(df_discharge[df_discharge["normcurrent"] != 0]["Current"]).mean()

    # Charge pulse
    df_charge = df_pulse[df_pulse["charge_pulse"] == 1]
    t4 = df_charge["TestTime"].iloc[0]
    V4 = df_charge["Voltage"].iloc[0]

    t5 = df_charge["TestTime"].iloc[-1]
    V5 = df_charge["Voltage"].iloc[-1]

    t3 = df[(df["TestTime"] < t4) & (df["normcurrent"] == 0)]["TestTime"].iloc[-1]
    V3 = df[df["TestTime"] == t2]["Voltage"].mean()
    Icharge = abs(df_charge[df_charge["normcurrent"] != 0]["Current"]).mean()

    return V0, V1, V2, V3, V4, V5, t0, t1, t2, t3, t4, t5, Idischarge, Icharge


def calculate_pulse_power_capability(Vmin, Vmax, V0, R_discharge, R_charge):
    """Calculates Pulse Power for HPPC test

    Parameters
    ----------
    Vmin : float
        Minimum voltage of the battery
    Vmax : float
        Maximum voltage of the battery
    V0 : float
        Open Circuit Voltage of the pulse
    R_discharge : float
        Discharge internal resistance of the pulse
    R_charge : float
        Charge internal resistance of the pulse

    Returns
    -------
    tuple
        Charge and discharge Pulse Power
    """
    P_discharge = Vmin * (V0 - Vmin) / R_discharge
    P_charge = Vmax * (Vmax - V0) / R_charge
    return P_discharge, P_charge
