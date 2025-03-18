import pandas as pd
import numpy as np

def pulse_number_GITT(df_input):
    """Select pulses for GITT data

    For every cycles, gives a pulse number for every current pulses.
    Then it gives this number for the last point before the pulse until the last point before the end of the relaxation
    Finally, it creates a nested Dataframe structuring the GITT data for each pulse
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the GITT data to process

    Returns
    -------
    Dict
        DataFrame containing the GITT parameters structured by pulse number
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


def global_calculation_GITT(df_nested):
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
        pulse_current = abs(df_pulse[df_pulse['Relaxation'] == 0]['Current']).mean()

        if len(df_pulse[df_pulse['Relaxation'] == 0]) > 2 and len(df_pulse[df_pulse['Relaxation'] == 1]) > 2 and df_pulse['not_pulse'].median() == 0:

            # print('pulse: ', pulse)

            resistance = calculate_ohmic_resistance(df_pulse, pulse_current)
            delta_Es = calculate_delta_Es(df_pulse)
            delta_Et = calculate_delta_Et(df_pulse)
            tau = calculate_tau(df_pulse)
            D = calculate_diffusion_coefficient(delta_Es, delta_Et, tau)

            R_1s = calculate_DCIR(df_pulse, 1)
            R_30s = calculate_DCIR(df_pulse, 30)
            R_60s = calculate_DCIR(df_pulse, 60)
            R_180s = calculate_DCIR(df_pulse, 178)


            df_coefficient = pd.DataFrame({'Pulse': pulse,
                            'Cycle': df_pulse['Cycle'].iloc[-1],
                            'TestTime': df_pulse['TestTime'].iloc[0],
                            'SOC': df_pulse['SOC'].iloc[0],
                            'Voltage': df_pulse['Voltage'].iloc[0],
                            'Diffusion Coefficient': D,
                            'Resistance': resistance,
                            'delta_Es': delta_Es,
                            'delta_Et': delta_Et,
                            'R_1s': R_1s,
                            'R_30s': R_30s,
                            'R_60s': R_60s,
                            'R_180s': R_180s,
                            'tau': tau,
                            'State': df_pulse['State'].iloc[0],},
                            index=['Pulse'])

            df_total = pd.concat([df_total, df_coefficient])
    return df_total


def calculate_ohmic_resistance(df_input, pulse_current):
    """Calculates the ohmic resistance for GITT test
    
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

    # Given that the drop start at 95% of the mean pulse current
    time_end = df_pulse[abs(df_pulse['Current']) >= 0.95 * pulse_current]['TestTime'].min()

    E1 = df_pulse['Voltage'].iloc[0]
    E2 = df_pulse.loc[df_pulse['TestTime'] >= time_end, 'Voltage'].iloc[0]

    resistance = abs((E2 - E1) / pulse_current)
    return resistance


def calculate_DCIR(df_input, resistance_time):
    """Calculates the DCIR for GITT test
    
    Parameters
    ----------
    df_input : pandas.DataFrame
        DataFrame containing the data corresponding to one pulse
    resistance_time : float
        Time after the beginning of the pulse at which the DCIR is calculated

    Returns
    -------
    float
        DCIR of the pulse
    """
    df_pulse = df_input[df_input['Relaxation'] == 0].copy()

    end_time = df_input['TestTime'].min() + resistance_time

    if end_time <= df_pulse['TestTime'].max():
        E1 = df_input['Voltage'].iloc[0]
        E2 = df_pulse.loc[df_pulse['TestTime'] >= end_time, 'Voltage'].iloc[0]
        resistance = abs((E2 - E1) / df_input['Current'].mean())
    else:
        resistance = np.nan

    return resistance
    

def calculate_delta_Es(df_input):
    """Calculates the delta_Es for GITT test
    
    Parameters
    ----------
    df_input : pandas.DataFrame
        DataFrame containing the data corresponding to one pulse

    Returns
    -------
    float
        delta_Es of the pulse
    """
    df_pulse = df_input.copy()

    Es1 = df_pulse['Voltage'].iloc[0]
    Es2 = df_pulse['Voltage'].iloc[-1]

    delta_Es = Es2 - Es1
    return delta_Es

def calculate_delta_Et(df_input):
    """Calculates the delta_Et for GITT test
    
    Parameters
    ----------
    df_input : pandas.DataFrame
        DataFrame containing the data corresponding to one pulse

    Returns
    -------
    float
        delta_Et of the pulse
    """
    df_current = df_input[(df_input['Relaxation'] == 0)].copy()

    Et1 = df_current['Voltage'].iloc[0]
    Et2 = df_current['Voltage'].iloc[-1]

    delta_Et = Et2 - Et1
    return delta_Et

def calculate_tau(df_pulse):
    """Calculates tau (Time duration of the pulse) for GITT test
    
    Parameters
    ----------
    df_pulse : pandas.DataFrame
        DataFrame containing the data corresponding to one pulse

    Returns
    -------
    float
        delta_Es of the selected pulse
    """
    df_current = df_pulse[df_pulse['Relaxation'] == 0].iloc[1:]

    tau = df_current['TestTime'].max() - df_current['TestTime'].min()
    return tau


def calculate_diffusion_coefficient(delta_Es, delta_Et, tau):
    """Calculates diffusion coefficient for GITT test
    
    Parameters
    ----------
    delta_Es : float
        delta_Es of the pulse
    delta_Et : float
        delta_Et of the pulse
    tau : float
        Time duration of the pulse

    Returns
    -------
    float
        Diffusion coefficient of the selected pulse
    """
    radius = 5e-6

    D = 4/(9 * np.pi) * radius**2 / tau * (delta_Es / delta_Et)**2
    return D
