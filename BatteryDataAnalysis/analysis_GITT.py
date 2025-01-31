import pandas as pd
import numpy as np

def pulse_number(df_input):
    # A pulse starts the time just before the pulse and ends the time just before the next pulse at the end of the relaxation time
    df = df_input.copy()

    df['is_transition'] = (df['Current'] != 0).astype(int).diff().fillna(0).gt(0).cumsum()
    df['Pulse'] = df['is_transition'].where(df['Current'] != 0).ffill().fillna(0).astype(int).shift(-1)
    df['Relaxation'] = (df['Current'] == 0).astype(int)
    
    df.drop(columns=['is_transition'], inplace=True)

    return df


def ohmic_resistance(df_input, pulse_current):
    df_pulse = df_input.copy()

    # Given that the drop start at 95% of the mean pulse current
    time_end = df_pulse[abs(df_pulse['Current'] * 1e-3) >= 0.95 * pulse_current]['TestTime'].min()

    E1 = df_pulse['Voltage'].iloc[0]
    E2 = df_pulse.loc[df_pulse['TestTime'] >= time_end, 'Voltage'].iloc[0]

    resistance = abs((E2 - E1) / pulse_current)
    return resistance
    

def delta_Es_calculation(df_input):
    df_pulse = df_input.copy()

    Es1 = df_pulse['Voltage'].iloc[0]
    Es2 = df_pulse['Voltage'].iloc[-1]

    delta_Es = Es2 - Es1
    return delta_Es

def delta_Et_calculation(df_input):
    df_current = df_input[(df_input['Relaxation'] == 0)].copy()

    Et1 = df_current['Voltage'].iloc[0]
    Et2 = df_current['Voltage'].iloc[-1]

    delta_Et = Et2 - Et1
    return delta_Et

def tau_calculation(df_pulse):
    df_current = df_pulse[df_pulse['Relaxation'] == 0].iloc[1:]

    tau = df_current['TestTime'].max() - df_current['TestTime'].min()
    return tau


def diffusion_coefficient1(delta_Es, delta_Et, tau):
    radius = 5e-6

    D = 4/(9 * np.pi) * radius**2 / tau * (delta_Es / delta_Et)**2
    return D

def diffusion_coefficient2(df_input, delta_Es, tau):
    radius = 5e-6

    df_pulse = df_input.copy()
    df_relax = df_pulse[(df_pulse['Relaxation'] == 1)].iloc[1:]

    time = np.sqrt(df_relax['TestTime'] - df_relax['TestTime'].min())

    start_time, end_time = time.min(), time.max()
    max_time = 0.1 * (end_time - start_time) + start_time

    voltage_min = df_relax.loc[time == start_time, 'Voltage'].iloc[0]
    voltage_max = df_relax.loc[time >= max_time, 'Voltage'].iloc[0]

    slope = (voltage_max - voltage_min) / (max_time - start_time)

    D = 4/(9 * np.pi) * (radius / tau)**2 * (delta_Es / slope)**2

    return D

def diffusion_coefficient3(df_input, delta_Es, tau):
    radius = 5e-6

    df_pulse = df_input.copy()
    df_relax = df_pulse[(df_pulse['Relaxation'] == 1)].iloc[1:]

    t_relax = df_relax['TestTime'] - df_relax['TestTime'].min()
    time = (t_relax + tau)**0.5 - (t_relax)**0.5
    
    start_time, end_time = time.min(), time.max()
    min_time = 0.05 * (end_time - start_time) + start_time

    voltage_max = df_relax.loc[time == start_time, 'Voltage'].iloc[0]
    voltage_min = df_relax.loc[time <= min_time, 'Voltage'].iloc[0]

    slope = (voltage_max - voltage_min) / (min_time - start_time)

    D = 4/(9 * np.pi) * (radius / tau)**2 * (delta_Es / slope)**2

    return D

def global_calculation(df):
    df_total = pd.DataFrame()
    for pulse in df['Pulse'].unique():
        print('Pulse: '+str(pulse))
        df_pulse = df[df['Pulse'] == pulse].copy()
        pulse_current = abs(df_pulse[df_pulse['Relaxation'] == 0]['Current']).mean() * 1e-3

        if len(df_pulse[df_pulse['Relaxation'] == 0]) >= 2 and len(df_pulse[df_pulse['Relaxation'] == 1]) >= 2:

            resistance = ohmic_resistance(df_pulse, pulse_current)
            delta_Es = delta_Es_calculation(df_pulse)
            delta_Et = delta_Et_calculation(df_pulse)
            tau = tau_calculation(df_pulse)
            D_method1 = diffusion_coefficient1(delta_Es, delta_Et, tau)
            D_method2 = diffusion_coefficient2(df_pulse, delta_Es, tau)
            D_method3 = diffusion_coefficient3(df_pulse, delta_Es, tau)

            relax_time = df_pulse[df_pulse['Relaxation'] == 0]['TestTime'].max()

            df_coefficient = pd.DataFrame({'Pulse': pulse,
                            'Cycle': df_pulse['Cycle'].iloc[0],
                            'SysTime': relax_time,
                            'SOC': df_pulse[df_pulse['TestTime'] == relax_time]['SOC'].mean(),
                            'D_method1': D_method1,
                            'D_method2': D_method2,
                            'D_method3': D_method3,
                            'Resistance': resistance,
                            'delta_Es': delta_Es,
                            'delta_Et': delta_Et,
                            'τ': tau,
                            'State': df_pulse['State'].iloc[0],},
                            index=['Pulse'])

            df_total = pd.concat([df_total, df_coefficient])

    return df_total