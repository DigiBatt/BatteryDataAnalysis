import pandas as pd
import numpy as np
from whittaker_eilers import WhittakerSmoother

def pulse_number_ICI(df_input):
    """Select pulses for ICI data

    For every cycles, gives a pulse number for every current pulses.
    Then it gives this number for the last point before the pulse until the last point before the end of the relaxation
    Finally, it creates a nested Dataframe structuring the ICI data for each pulse
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the ICI data to process

    Returns
    -------
    Dict
        DataFrame containing the ICI parameters structured by pulse number
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


def global_calculation_ICI(df_nested):
    """Calculates the ICI parameters

    Given a nested DataFrame containing the ICI data, this function calculates the ICI parameters for each pulse
    
    Parameters
    ----------
    df_nested : Dict
        Dict containing the nested DataFrames corresponding to each pulse

    Returns
    -------
    pandas.DataFrame
        Dict containing the nested DataFrames as values and their pulse number as keys
    """
    df_total = pd.DataFrame()
    for pulse, df_pulse in df_nested.items():
        pulse_current = abs(df_pulse[df_pulse['Relaxation'] == 0]['Current']).mean()

        if len(df_pulse[df_pulse['Relaxation'] == 0]) > 5 and len(df_pulse[df_pulse['Relaxation'] == 1]) > 5:

            resistance = calculate_ohmic_resistance(df_pulse, pulse_current)
            delta_Es = calculate_delta_Es(df_pulse)
            delta_Et = calculate_delta_Et(df_pulse)
            tau = calculate_tau(df_pulse)
            D = calculate_diffusion_coefficient(delta_Es, delta_Et, tau)

            df_coefficient = pd.DataFrame({'Pulse': pulse,
                            'Cycle': df_pulse['Cycle'].iloc[0],
                            'TestTime': df_pulse['TestTime'].iloc[0],
                            'SOC': df_pulse['SOC'].iloc[0],
                            'Voltage': df_pulse['Voltage'].iloc[0],
                            'Diffusion Coefficient': D,
                            'Resistance': resistance,
                            'tau': tau,
                            'State': df_pulse['State'].iloc[0],},
                            index=['Pulse'])

            df_total = pd.concat([df_total, df_coefficient])
    return df_total


def calculate_ohmic_resistance(df_input, pulse_current):
    """Calculates the ohmic resistance for ICI test
    
    Parameters
    ----------
    df_input : pandas.DataFrame
        DataFrame containing the data corresponding to one pulse
    pulse_current : float
        Mean current during the pulse

    Returns
    -------
    float
        Ohmic resistance of the pulse
    """
    df_pulse = df_input.copy()
    df_relax = df_pulse[(df_pulse['Relaxation'] == 1)].copy()

    E1 = df_pulse['Voltage'].iloc[0]
    E2 = df_relax['Voltage'].iloc[0]

    resistance = abs((E2 - E1) / pulse_current)
    return resistance
    

def calculate_delta_Es(df_input):
    """Calculates delta_Es for ICI test
    
    Parameters
    ----------
    df_input : pandas.DataFrame
        DataFrame containing the data corresponding to one pulse

    Returns
    -------
    float
        delta_Es of the selected pulse
    """
    df_pulse = df_input.copy()

    Es1 = df_pulse['Voltage'].iloc[0]
    Es2 = df_pulse['Voltage'].iloc[-1]

    t1 = df_pulse[df_pulse['Relaxation'] == 1]['TestTime'].iloc[-1]
    t2 = df_pulse['TestTime'].iloc[-1]

    delta_Es = Es2 - Es1
    return delta_Es / (t2 - t1)

def calculate_delta_Et(df_input):
    """Calculates delta_Et for ICI test
    
    Parameters
    ----------
    df_input : pandas.DataFrame
        DataFrame containing the data corresponding to one pulse

    Returns
    -------
    float
        delta_Et of the selected pulse
    """
    df_relax = df_input[(df_input['Relaxation'] == 1)].copy()
    
    signal_noise = np.diff(df_relax['Voltage'], 2)
    signal_noise_absolute = np.abs(signal_noise).reshape(-1, 1)
    signal_noise_normalized = signal_noise_absolute / abs(df_relax['Voltage']).max()
    weights = np.exp(-signal_noise_normalized)
    weights_padded = np.concatenate(([weights[0]], weights, [weights[-1]]))

    whittaker_smoother = WhittakerSmoother(lmbda=1e3, order=1, data_length=len(df_relax), x_input=df_relax['TestTime'], weights=weights_padded)
    df_relax['Voltage'] = whittaker_smoother.smooth(df_relax['Voltage'].values)

    # delta_E = abs(df_current['Voltage']).diff().iloc[0:5]
    # delta_t = np.sqrt((df_current['TestTime'] - df_current['TestTime'].iloc[0])).diff().iloc[0:5]

    delta_E = df_relax['Voltage'].iloc[-1] - df_relax['Voltage'].iloc[0]
    delta_t = np.sqrt((df_relax['TestTime'].iloc[-1] - df_relax['TestTime'].iloc[0]))

    delta_Et = delta_E / delta_t

    delta_Et = (df_relax['Voltage'].diff() / df_relax['TestTime'].diff()).median()
    return delta_Et


def calculate_tau(df_pulse):
    """Calculates tau (Time duration of the pulse) for ICI test
    
    Parameters
    ----------
    df_pulse : pandas.DataFrame
        DataFrame containing the data corresponding to one pulse

    Returns
    -------
    float
        delta_Es of the selected pulse
    """
    df_current = df_pulse[df_pulse['Relaxation'] == 1].iloc[1:]

    tau = df_current['TestTime'].max() - df_current['TestTime'].min()
    return tau


def calculate_diffusion_coefficient(delta_Es, delta_Et):
    """Calculates diffusion coefficient for ICI test
    
    Parameters
    ----------
    delta_Es : float
        delta_Es of the pulse
    delta_Et : float
        delta_Et of the pulse

    Returns
    -------
    float
        Diffusion coefficient of the selected pulse
    """
    cst = 1e-5

    D = 4/np.pi * (delta_Es / delta_Et)**2 * cst**2
    return D