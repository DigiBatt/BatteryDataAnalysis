import pandas as pd
import numpy as np

def pulse_number_HPPC(df_input):
    """Select pulses for HPPC data

    For every cycles, gives a pulse number for every HPPC pattern (containing a charge and discharge pulse followed by a relaxation time).
    Then it gives this number for the last point before the first pulse until the last point before the end of the relaxation
    Finally, it creates a nested Dataframe structuring the HPPC data for each pattern (called 'Pulse' to be consistent with GITT and ICI processing)
    
    Parameters
    ----------
    df_input : pandas.DataFrame
        DataFrame containing the HPPC data to process

    Returns
    -------
    Dict
        Dict containing the nested DataFrames as values and their pulse number as keys
    """
    # A pulse starts the time just before the pulse and ends the time just before the next pulse at the end of the relaxation time
    df = df_input.copy()

    pulse_count = 0
    for cycle in df['Cycle'].unique():
        for state in df[df['Cycle'] == cycle]['State'].unique():
            df_state = df[(df['State'] == state) & (df['Cycle'] == cycle)].copy()

            # Count the number of positive and negative current pulses (including relaxation)
            df_state['raw_neg_pulse_count'] = (df_state['normcurrent'] < 0).astype(int).diff().fillna(0).gt(0).cumsum().ffill().shift(-1)
            df_state['raw_pos_pulse_count'] = (df_state['normcurrent'] > 0).astype(int).diff().fillna(0).gt(0).cumsum().ffill().shift(-1)
            df_state = df_state.dropna(subset=['raw_neg_pulse_count', 'raw_pos_pulse_count'])

            # If charging the only negative current is the HPPC pulse
            if state == 'C':
                df['discharge_pulse'] = (df['normcurrent'] < 0).astype(int)
                
                df_group = df_state[df_state['normcurrent'] < 0].groupby('raw_neg_pulse_count')['TestTime'].agg(lambda x: x.max() - x.min())
                pulse_time = df_group.median()

                # If a positive pulse is the same length as the negative pulse then it is the HPPC pulse (and not the relaxation)
                for pulse in df_state['raw_pos_pulse_count'].unique():
                    df_pulse = df_state[(df_state['raw_pos_pulse_count'] == pulse) & (df_state['normcurrent'] > 0)]
                    if df_pulse['TestTime'].max() - df_pulse['TestTime'].min() < pulse_time * 1.5:
                        df_state.loc[(df_state['raw_pos_pulse_count'] == pulse) & (df_state['normcurrent'] > 0), 'charge_pulse'] = 1
                        pulse_count += 1

                    df_state.loc[df_state['raw_pos_pulse_count'] == pulse, 'pos_pulse_count'] = pulse_count

                df_state['pos_pulse_count'] = df_state['pos_pulse_count'].ffill()
                df_state['Pulse'] = np.maximum(df_state['raw_neg_pulse_count'], df_state['pos_pulse_count'])

            # If discharging the only positive current is the HPPC pulse
            elif state == 'D':
                df_state['charge_pulse'] = (df_state['normcurrent'] > 0).astype(int)

                df_group = df_state[df_state['normcurrent'] > 0].groupby('raw_neg_pulse_count')['TestTime'].agg(lambda x: x.max() - x.min())
                pulse_time = df_group.median()

                # If a negative pulse is the same length as the positive pulse then it is the HPPC pulse (and not the relaxation)
                for pulse in df_state['raw_neg_pulse_count'].unique():
                    df_pulse = df_state[(df_state['raw_neg_pulse_count'] == pulse) & (df_state['normcurrent'] < 0)]
                    if df_pulse['TestTime'].max() - df_pulse['TestTime'].min() < pulse_time * 1.5:
                        df_state.loc[(df_state['raw_neg_pulse_count'] == pulse) & (df_state['normcurrent'] < 0), 'discharge_pulse'] = 1
                        pulse_count += 1

                    df_state.loc[df_state['raw_neg_pulse_count'] == pulse, 'neg_pulse_count'] = pulse_count

                df_state['neg_pulse_count'] = df_state['neg_pulse_count'].ffill()
                df_state['Pulse'] = np.maximum(df_state['neg_pulse_count'], df_state['raw_pos_pulse_count'])


            df.loc[(df['State'] == state) & (df['Cycle'] == cycle), 'Pulse'] = df_state['Pulse'].astype(int)
            df.loc[(df['State'] == state) & (df['Cycle'] == cycle), 'charge_pulse'] = df_state['charge_pulse'].fillna(0)
            df.loc[(df['State'] == state) & (df['Cycle'] == cycle), 'discharge_pulse'] = df_state['discharge_pulse'].fillna(0)

    df = df.dropna(subset=['Pulse'])
    df_nested = {int(pulse): df[df['Pulse'] == pulse] for pulse in df['Pulse'].unique()}

    return df_nested


def global_calculation_HPPC(df_nested):
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
    Vmin = df['Voltage'].min()
    Vmax = df['Voltage'].max()

    for pulse, df_pulse in df_nested.items():
        # print('Pulse: '+str(pulse))

        df_charge = df_pulse[df_pulse['charge_pulse'] == 1]
        df_discharge = df_pulse[df_pulse['discharge_pulse'] == 1]

        OCV = df_pulse['Voltage'].iloc[0]

        if len(df_charge) >= 2 and len(df_discharge) >= 2 and pulse != 0:

            R_discharge, R_charge = calculate_internal_resistance(df_pulse, df_charge, df_discharge)
            P_discharge, P_charge = calculate_pulse_power_capability(Vmin, Vmax, OCV, R_discharge, R_charge)

            df_coefficient = pd.DataFrame({'Pulse': pulse,
                                           'Cycle': df_pulse['Cycle'].iloc[0],
                                           'TestTime': df_pulse['TestTime'].iloc[0],
                                           'SOC': df_pulse['SOC'].iloc[0],
                                           'Voltage': OCV,
                                           'State': df_pulse['State'].iloc[0],
                                           'R_charge': R_charge,
                                           'R_discharge': R_discharge,
                                           'P_charge': P_charge,
                                           'P_discharge': P_discharge,
                                           },
                                           index=['Pulse'])

            df_total = pd.concat([df_total, df_coefficient])
    return df_total



def calculate_internal_resistance(df_pulse, df_charge, df_discharge):
    """Calculates internal resistance for HPPC test
    
    Parameters
    ----------
    df_input : pandas.DataFrame
        DataFrame containing the data corresponding to one HPPC pattern
    df_charge : pandas.DataFrame
        DataFrame containing the data corresponding to the charge pulse
    df_discharge : pandas.DataFrame
        DataFrame containing the data corresponding to the discharge pulse

    Returns
    -------
    tuple(float, float)
        Charge and discharge internal resistance
    """
    df = df_pulse.copy()

    discharge_start = df_discharge['TestTime'].iloc[0]
    t0 = df[(df['TestTime'] < discharge_start) & (df['normcurrent'] == 0)]['TestTime'].iloc[-1]
    v0 = df[df['TestTime'] == t0]['Voltage'].mean()
    i0 = df[df['TestTime'] == t0]['Current'].mean()

    t1 = df_discharge['TestTime'].iloc[-1]
    v1 = df[df['TestTime'] == t1]['Voltage'].mean()
    i1 = df[df['TestTime'] == t1]['Current'].mean()

    charge_start = df_charge['TestTime'].iloc[0]
    t2 = df[(df['TestTime'] < charge_start) & (df['normcurrent'] == 0)]['TestTime'].iloc[-1]
    v2 = df[df['TestTime'] == t2]['Voltage'].mean()
    i2 = df[df['TestTime'] == t2]['Current'].mean()

    t3 = df_charge['TestTime'].iloc[-1]
    v3 = df[df['TestTime'] == t3]['Voltage'].mean()
    i3 = df[df['TestTime'] == t3]['Current'].mean()

    R_discharge = abs((v1 - v0) / (i0 - i1)) 
    R_charge = abs((v3 - v2) / (i3 - i2))

    return R_discharge, R_charge


def calculate_pulse_power_capability(Vmin, Vmax, OCV, R_discharge, R_charge):
    """Calculates internal resistance for HPPC test
    
    Parameters
    ----------
    Vmin : float
        Minimum voltage of the pulse
    Vmax : float
        Maximum voltage of the pulse
    OCV : float
        Open circuit voltage
    R_discharge : float
        Discharge internal resistance
    R_charge : float
        Charge internal resistance

    Returns
    -------
    tuple(float, float)
        Charge and discharge Pulse Power
    """
    P_discharge = Vmin * (OCV - Vmin) / R_discharge 
    P_charge = Vmax * (Vmax - OCV) / R_charge 
    return P_discharge, P_charge

