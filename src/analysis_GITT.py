import pandas as pd
import numpy as np
from create_custom_functions import add_function
from whittaker_eilers import WhittakerSmoother
from scipy.interpolate import interp1d

def pulse_number_GITT(df_input):
    """Pulse selection for GITT data

    - For every cycles, gives a pulse number for every current pulses.
    - Then it gives this number for the last point before the pulse until the last point before the end of the relaxation.
    - Finally, it creates a nested Dataframe structuring the GITT data for each pulse.

    Here is a schema of a selected pulse and its relevant points:

    .. image:: ../../../../_static/pulse_GITT.png
        :width: 2000px
        :align: center
        :alt: Example of a pulse selected by the function for GITT data

    The corresponding parameters that can be used in a custom function (see :func:`add_function`) are: 
    *V0, V1, V2, V3, t0, t1, t2, t3, Ipulse, delta_Eocv, delta_Edrop, delta_Epulse* that are calculated in :func:`calculate_relevant_points_GITT`.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the GITT data to process

    Returns
    -------
    dict
        Dict containing the nested DataFrames as values and their pulse number as keys
    """
    # A pulse starts the time just before the pulse and ends the time just before the next pulse at the end of the relaxation time
    df = df_input.copy()

    df['Pulse'] = (df['normcurrent'] != 0).astype(int).diff().fillna(0).gt(0).cumsum().ffill().shift(-1)
    df['Relaxation'] = (df['normcurrent'] == 0).astype(int)
    df['not_pulse'] = 0

    for cycle in df['Cycle'].unique():
        for state in df[df['Cycle'] == cycle]['State'].unique():
            df_state = df[(df['State'] == state) & (df['Cycle'] == cycle)].copy()

            df_group = df_state[df_state['normcurrent'] != 0].groupby('Pulse')['TestTime'].agg(lambda x: x.max() - x.min())
            pulse_time = df_group.median()

            for pulse in df_state['Pulse'].unique():
                df_pulse = df_state[(df_state['Pulse'] == pulse) & (df_state['normcurrent'] != 0)]
                if df_pulse['TestTime'].max() - df_pulse['TestTime'].min() > pulse_time * 3:
                    df.loc[(df['State'] == state) & (df['Cycle'] == cycle) & (df['Pulse'] == pulse), 'not_pulse'] = 1

    df = df.dropna(subset=['Pulse'])
    df_nested = {int(pulse): df[df['Pulse'] == pulse] for pulse in df['Pulse'].unique()}

    return df_nested


def global_calculation_GITT(df_nested, my_func_list):
    """Calculates the GITT parameters

    Given a nested DataFrame containing the GITT data, this function calculates the GITT parameters for each pulse
    
    Parameters
    ----------
    df_nested : Dict
        Dict containing the nested DataFrames corresponding to each pulse

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the GITT parameters structured by pulse number
    """
    df_total = pd.DataFrame()
    for pulse, df_pulse in df_nested.items():
        enough_relaxation_data = len(df_pulse[df_pulse['Relaxation'] == 0]) > 2
        enough_pulse_data = len(df_pulse[df_pulse['Relaxation'] == 1]) > 2
        is_a_pulse = df_pulse['not_pulse'].median() == 0
        if enough_relaxation_data and enough_pulse_data and is_a_pulse:

            try:
                V0, V1, V2, V3, t0, t1, t2, t3, Ipulse, delta_Eocv, delta_Edrop, delta_Epulse = calculate_relevant_points_GITT(df_pulse)
                Voltage = interp1d(df_pulse['TestTime'], df_pulse['Voltage'], kind='linear')
                Current = interp1d(df_pulse['TestTime'], df_pulse['Current'], kind='linear')
                Capacity = interp1d(df_pulse['TestTime'], df_pulse['Capacity'], kind='linear')

                tau = t2 - t1
                resistance = abs((V1 - V0) / Ipulse)
                D = calculate_diffusion_coefficient_GITT(delta_Eocv, delta_Epulse, tau)

                R_1s = calculate_DCIR(V0, Voltage, t0, t3, Ipulse, 1)
                R_30s = calculate_DCIR(V0, Voltage, t0, t3, Ipulse, 30)
                R_60s = calculate_DCIR(V0, Voltage, t0, t3, Ipulse, 60)
                R_180s = calculate_DCIR(V0, Voltage, t0, t3, Ipulse, 178)

                df_coefficient = pd.DataFrame({'Pulse': pulse,
                                'Cycle': df_pulse['Cycle'].iloc[-1],
                                'State': df_pulse['State'].iloc[-1],
                                'TestTime': t0,
                                'SOC': df_pulse['SOC'].iloc[0],
                                'OCV': V0,
                                'Ipulse': Ipulse,
                                'Diffusion Coefficient': D,
                                'Resistance': resistance,
                                # 'R_1s': R_1s,
                                # 'R_30s': R_30s,
                                # 'R_60s': R_60s,
                                # 'R_180s': R_180s,
                                },
                                index=['Pulse'])

                # Parameters that can be used in an external function
                kwargs = {
                    'V0': V0,
                    'V1': V1,
                    'V2': V2,
                    'V3': V3,
                    't0': t0,
                    't1': t1,
                    't2': t2,
                    't3': t3,
                    'Ipulse': Ipulse,
                    'delta_Eocv': delta_Eocv,
                    'delta_Edrop': delta_Edrop,
                    'delta_Epulse': delta_Epulse,
                    'Voltage': Voltage,
                    'Current': Current,
                    'Capacity': Capacity,
                    'df_pulse': df_pulse,
                    'State': df_pulse['State'].iloc[0],
                }

                for func in my_func_list:
                    var_name = func.__name__
                    new_var = add_function(func, **kwargs)
                    if new_var is not None:
                        df_coefficient[var_name] = new_var

            except Exception as e:
                print(f'Error processing pulse {pulse}: {e}')
        

            df_total = pd.concat([df_total, df_coefficient])
    return df_total


def calculate_relevant_points_GITT(df_pulse):
    """Calculates the relevant points represented in the :func:`pulse_number_GITT` documentation.  

    These parameters can be used in a custom function to calculate new parameters (see :func:`add_function`).
    
    Parameters
    ----------
    df_pulse : pandas.DataFrame
        DataFrame containing the data corresponding to one pulse

    Returns
    -------
    tuple
        Tuple containing the relevant points : V0, V1, V2, V3, t0, t1, t2, t3, Ipulse, delta_Eocv, delta_Edrop, delta_Epulse
    """
    df_current = df_pulse[(df_pulse['Relaxation'] == 0)].copy()

    # Relaxation Voltage smoothing
    signal_noise = np.diff(df_current['Voltage'], 2)
    signal_noise_absolute = np.abs(signal_noise).reshape(-1, 1)
    signal_noise_normalized = signal_noise_absolute / abs(df_current['Voltage']).max()
    weights = np.exp(-signal_noise_normalized)
    weights_padded = np.concatenate(([weights[0]], weights, [weights[-1]]))
    whittaker_smoother = WhittakerSmoother(lmbda=1e3, order=1, data_length=len(df_current), x_input=df_current['TestTime'], weights=weights_padded)
    df_current['Voltage'] = whittaker_smoother.smooth(df_current['Voltage'].values)

    # Relevant points calculation
    V0 = df_pulse['Voltage'].iloc[0]
    V1 = df_current['Voltage'].iloc[0]
    V2 = df_current['Voltage'].iloc[-1]
    V3 = df_pulse['Voltage'].iloc[-1]

    t0 = df_pulse['TestTime'].iloc[0]
    t1 = df_current['TestTime'].iloc[0]
    t2 = df_current['TestTime'].iloc[-1]
    t3 = df_pulse['TestTime'].iloc[-1]

    Ipulse = df_pulse[df_pulse['normcurrent'] != 0]['Current'].mean()

    delta_Eocv = V3 - V0
    delta_Edrop = V1 - V0
    delta_Epulse = V2 - V1

    return V0, V1, V2, V3, t0, t1, t2, t3, Ipulse, delta_Eocv, delta_Edrop, delta_Epulse


def calculate_DCIR(V0, Voltage, t0, t3, Ipulse, r_time):
    """Calculates the DCIR for GITT test
    
    Parameters
    ----------
    V0 : float
    Voltage : interp1d
        Interpolation function of the voltage
    t0 : float
    t3 : float
    Ipulse : float
    r_time : float
        Time after which DCIR is calculated

    Returns
    -------
    float
        DCIR of the pulse
    """
    if t0 + r_time <= t3:
        V_r_time = Voltage(t0 + r_time)
        resistance = abs((V_r_time - V0) / Ipulse)
    else:
        resistance = np.nan

    return resistance


def calculate_diffusion_coefficient_GITT(delta_Eocv, delta_Epulse, tau):
    r"""Calculates diffusion coefficient for GITT test

    See the :func:`pulse_number_GITT` documentation for the points definition.  
    The formula comes from the paper from T. Schied et al. (2021) 
    *"Determining the Diffusion Coefficient of Lithium Insertion Cathodes from GITT measurements: Theoretical Analysis for low Temperatures"*

    The diffusion coefficient is calculated using the following formula:

    .. math::

        D = \frac{4}{9 \pi} \cdot \frac{\text{radius}^2}{\tau} \cdot \left(\frac{\Delta E_{OCV}}{\Delta E_{pulse}}\right)^2
    
    Parameters
    ----------
    V0 : float
    V1 : float
    V2 : float
    V3 : float
    tau : float
        Time duration of the pulse

    Returns
    -------
    float
        Diffusion coefficient of the selected pulse
    """
    radius = 5e-6

    D = 4/(9 * np.pi) * radius**2 / tau * (delta_Eocv / delta_Epulse)**2
    return D
