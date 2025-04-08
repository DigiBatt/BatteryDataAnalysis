import pandas as pd
import numpy as np
from create_custom_functions import add_function
from whittaker_eilers import WhittakerSmoother
from scipy.interpolate import interp1d

def pulse_number_ICI(df_input):
    """Pulse selection for ICI data

    - For every cycles, it gives a pulse number for every relaxation pulse.
    - Then it gives this number for the last point before the pulse until the last point before the end of the charging/discharging.
    - Finally, it creates a nested Dataframe structuring the ICI data for each pulse.

    Here is a schema of a selected pulse and its relevant points:

    .. image:: ../../../../_static/pulse_ICI.png
        :width: 2000px
        :align: center
        :alt: Example of a pulse selected by the function for ICI data

    .. image:: ../../../../_static/pulse_ICI_relaxation.png
        :width: 2000px
        :align: center
        :alt: Relaxation part of a pulse selected by the function for ICI data

    The corresponding parameters that can be used in a custom function (see :func:`add_function`) are:
    *V0, V1, V2, V3, t0, t1, t2, t3, Ipulse, delta_Ec, delta_Edrop, delta_Epulse* that are calculated in :func:`calculate_relevant_points_ICI`.
        
    Parameters
    ----------
    df : pandas.DataFrame
       DataFrame containing the ICI data to process

    Returns
    -------
    Dict containing the nested DataFrames as values and their pulse number as keys
    """
    # A pulse starts the time just before the pulse and ends the time just before the next pulse at the end of the relaxation time
    df = df_input.copy()

    df['Pulse'] = (df['normcurrent'] == 0).astype(int).diff().fillna(0).gt(0).cumsum().ffill().shift(-1)
    df['Relaxation'] = (df['normcurrent'] == 0).astype(int)
    df['not_pulse'] = 0

    for cycle in df['Cycle'].unique():
        for state in df[df['Cycle'] == cycle]['State'].unique():
            df_state = df[(df['State'] == state) & (df['Cycle'] == cycle)].copy()

            df_group = df_state[df_state['normcurrent'] == 0].groupby('Pulse')['TestTime'].agg(lambda x: x.max() - x.min())
            pulse_time = df_group.median()

            for pulse in df_state['Pulse'].unique():
                df_pulse = df_state[(df_state['Pulse'] == pulse) & (df_state['normcurrent'] == 0)]
                if df_pulse['TestTime'].max() - df_pulse['TestTime'].min() > pulse_time * 3:
                    df.loc[(df['State'] == state) & (df['Cycle'] == cycle) & (df['Pulse'] == pulse), 'not_pulse'] = 1

    df = df.dropna(subset=['Pulse'])
    df_nested = {int(pulse): df[df['Pulse'] == pulse] for pulse in df['Pulse'].unique()}

    return df_nested


def global_calculation_ICI(df_nested, my_func_list):
    """Calculates the ICI parameters

    Given a nested DataFrame containing the ICI data, this function calculates the ICI parameters for each pulse
    
    Parameters
    ----------
    df_nested : Dict
        Dict containing the nested DataFrames corresponding to each pulse

    Returns
    -------
    dict
        Dict containing the nested DataFrames as values and their pulse number as keys
    """
    df_total = pd.DataFrame()
    for pulse, df_pulse in df_nested.items():

        if len(df_pulse[df_pulse['Relaxation'] == 0]) > 5 and len(df_pulse[df_pulse['Relaxation'] == 1]) > 5:
            try:
                V0, V1, V2, V3, t0, t1, t2, t3, Ipulse, delta_Ec, delta_Edrop, delta_Epulse = calculate_relevant_points_ICI(df_pulse)

                tau = t2 - t1
                resistance = abs((V1 - V0) / Ipulse)
                D = calculate_diffusion_coefficient_ICI(t1, t2, t3, delta_Ec, delta_Epulse)

                df_coefficient = pd.DataFrame({'Pulse': pulse,
                                'Cycle': df_pulse['Cycle'].iloc[-1],
                                'State': df_pulse['State'].iloc[-1],
                                'TestTime': df_pulse['TestTime'].iloc[0],
                                'SOC': df_pulse['SOC'].iloc[0],
                                'OCV': df_pulse['Voltage'].iloc[0],
                                'Diffusion Coefficient': D,
                                'Resistance': resistance,
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
                    'delta_Ec': delta_Ec,
                    'delta_Edrop': delta_Edrop,
                    'delta_Epulse': delta_Epulse,
                    'Voltage': interp1d(df_pulse['TestTime'], df_pulse['Voltage'], kind='linear'),
                    'Current': interp1d(df_pulse['TestTime'], df_pulse['Current'], kind='linear'),
                    'Capacity': interp1d(df_pulse['TestTime'], df_pulse['Capacity'], kind='linear'),
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


def calculate_relevant_points_ICI(df_pulse):
    """Calculates the relevant points represented in the :func:`pulse_number_ICI` documentation.  

    These parameters can be used in a custom function to calculate new parameters (see :func:`add_function`).
    
    Parameters
    ----------
    df_pulse : pandas.DataFrame
        DataFrame containing the data corresponding to one pulse

    Returns
    -------
    tuple
        Tuple containing the relevant points : V0, V1, V2, V3, t0, t1, t2, t3, Ipulse, delta_Ec, delta_Edrop, delta_Epulse
    """
    df_relax = df_pulse[(df_pulse['Relaxation'] == 1)].copy()

    # Relaxation Voltage smoothing
    signal_noise = np.diff(df_relax['Voltage'], 2)
    signal_noise_absolute = np.abs(signal_noise).reshape(-1, 1)
    signal_noise_normalized = signal_noise_absolute / abs(df_relax['Voltage']).max()
    weights = np.exp(-signal_noise_normalized)
    weights_padded = np.concatenate(([weights[0]], weights, [weights[-1]]))
    whittaker_smoother = WhittakerSmoother(lmbda=1e3, order=1, data_length=len(df_relax), x_input=df_relax['TestTime'], weights=weights_padded)
    df_relax['Voltage'] = whittaker_smoother.smooth(df_relax['Voltage'].values)

    # Relevant points calculation
    V0 = df_pulse['Voltage'].iloc[0]
    V1 = df_relax['Voltage'].iloc[0]
    V2 = df_relax['Voltage'].iloc[-1]
    V3 = df_pulse['Voltage'].iloc[-1]

    t0 = df_pulse['TestTime'].iloc[0]
    t1 = df_relax['TestTime'].iloc[0]
    t2 = df_relax['TestTime'].iloc[-1]
    t3 = df_pulse['TestTime'].iloc[-1]

    Ipulse = df_pulse[df_pulse['normcurrent'] != 0]['Current'].mean()

    delta_Ec = V3 - V0
    delta_Edrop = V1 - V0
    delta_Epulse = V2 - V1

    return V0, V1, V2, V3, t0, t1, t2, t3, Ipulse, delta_Ec, delta_Edrop, delta_Epulse


def calculate_diffusion_coefficient_ICI(t1, t2, t3, delta_Ec, delta_Epulse):
    r"""Calculates diffusion coefficient for ICI test

    See the :func:`pulse_number_ICI` documentation for the points definition.  
    The formula comes from the paper from Y. Chien et al. (2023) 
    *"Rapid determination of solid-state diffusion coefficients in Li-based batteries via intermittent current interruption method"*

    The diffusion coefficient is calculated using the following formula:

    .. math::

        D = \frac{4}{9\pi} \cdot \text{radius}^2 \cdot \left( \frac{\frac{\Delta E_c}{t_3 - t_2}}{\frac{\Delta E_{\text{pulse}}}{\sqrt{t_2 - t_1}}} \right)^2

    Parameters
    ----------
    t1 : float
    t2 : float
    t3 : float
    delta_Ec : float
    delta_Epulse : float

    Returns
    -------
    float
        Diffusion coefficient of the selected pulse
    """
    radius = 5e-6

    D = 4/(9 * np.pi) * radius**2 * ((delta_Ec / (t3 - t2)) / (delta_Epulse / (np.sqrt(t2 - t1))))**2
    return D