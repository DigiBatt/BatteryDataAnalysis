import pandas as pd
import numpy as np

def pulse_number_GITT(df_input):
    # A pulse starts the time just before the pulse and ends the time just before the next pulse at the end of the relaxation time
    df = df_input.copy()
    # df['Current_rounded'] = round(df['Current'], 1)
    df['Current_rounded'] = df['Current']

    df['Pulse'] = (df['Current_rounded'] != 0).astype(int).diff().fillna(0).gt(0).cumsum().ffill().shift(-1)
    df['Relaxation'] = (df['Current_rounded'] == 0).astype(int)

    df = df.drop(columns=['Current_rounded']).dropna(subset=['Pulse'])
    df_nested = {int(pulse): df[df['Pulse'] == pulse] for pulse in df['Pulse'].unique()}

    return df_nested


def ohmic_resistance(df_input, pulse_current):
    df_pulse = df_input.copy()

    # Given that the drop start at 95% of the mean pulse current
    time_end = df_pulse[abs(df_pulse['Current']) >= 0.95 * pulse_current]['TestTime'].min()

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


def diffusion_coefficient(delta_Es, delta_Et, tau):
    radius = 5e-6

    D = 4/(9 * np.pi) * radius**2 / tau * (delta_Es / delta_Et)**2
    return D


def global_calculation_GITT(df_nested):
    df_total = pd.DataFrame()
    for pulse, df_pulse in df_nested.items():
        pulse_current = abs(df_pulse[df_pulse['Relaxation'] == 0]['Current']).mean()

        if len(df_pulse[df_pulse['Relaxation'] == 0]) >= 2 and len(df_pulse[df_pulse['Relaxation'] == 1]) >= 2:

            resistance = ohmic_resistance(df_pulse, pulse_current)
            delta_Es = delta_Es_calculation(df_pulse)
            delta_Et = delta_Et_calculation(df_pulse)
            tau = tau_calculation(df_pulse)
            D = diffusion_coefficient(delta_Es, delta_Et, tau)


            df_coefficient = pd.DataFrame({'Pulse': pulse,
                            'Cycle': df_pulse['Cycle'].iloc[0],
                            'TestTime': df_pulse['TestTime'].iloc[0],
                            'SOC': df_pulse['SOC'].iloc[0],
                            'Voltage': df_pulse['Voltage'].iloc[0],
                            'Diffusion Coefficient': D,
                            'Resistance': resistance,
                            'delta_Es': delta_Es,
                            'delta_Et': delta_Et,
                            'tau': tau,
                            'State': df_pulse['State'].iloc[0],},
                            index=['Pulse'])

            df_total = pd.concat([df_total, df_coefficient])
    return df_total