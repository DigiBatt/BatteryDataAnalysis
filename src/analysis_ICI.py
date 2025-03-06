import pandas as pd
import numpy as np

def pulse_number_ICI(df_input):
    # A pulse starts the time just before the pulse and ends the time just before the next pulse at the end of the relaxation time
    df = df_input.copy()

    df['Pulse'] = (df['normcurrent'] == 0).astype(int).diff().fillna(0).gt(0).cumsum().ffill().shift(-1)
    df['Relaxation'] = (df['normcurrent'] == 0).astype(int)

    # for pulse in df['Pulse'].unique():
    #     start_time = df[df['Pulse'] == pulse]['TestTime'].min()
    #     end_time = df[df['Pulse'] == pulse]['TestTime'].max()

    #     if not df.loc[(df['TestTime'] <= start_time) & (df['normcurrent'] == 0), 'TestTime'].empty:
    #         pulse_start = df.loc[(df['TestTime'] <= start_time) & (df['normcurrent'] == 0), 'TestTime'].iloc[-1]
    #         df.loc[(df['TestTime'] >= pulse_start) & (df['TestTime'] <= end_time), 'Pulse'] = pulse

    # for pulse in df['Pulse'].unique():
    #     df.loc[df['Pulse'] == pulse, 'State'] = df[df['Pulse'] == pulse]['State'].mode()[0]
    #     df.loc[df['Pulse'] == pulse, 'Cycle'] = df[df['Pulse'] == pulse]['Cycle'].mode()[0]

    df = df.dropna(subset=['Pulse'])
    df_nested = {int(pulse): df[df['Pulse'] == pulse] for pulse in df['Pulse'].unique()}

    return df_nested


# def ohmic_resistance(df_input, pulse_current):
#     df_pulse = df_input.copy()

#     # Given that the drop start at 95% of the mean pulse current
#     time_end = df_pulse[abs(df_pulse['Current']) >= 0.95 * pulse_current]['TestTime'].min()

#     E1 = df_pulse['Voltage'].iloc[0]
#     E2 = df_pulse.loc[df_pulse['TestTime'] >= time_end, 'Voltage'].iloc[0]

#     resistance = abs((E2 - E1) / pulse_current)
#     return resistance
    

def delta_Es_calculation(df_input):
    df_pulse = df_input.copy()

    Es1 = df_pulse['Voltage'].iloc[0:5].median()
    Es2 = df_pulse['Voltage'].iloc[-1:-6:-1].median()

    delta_Es = Es2 - Es1
    return delta_Es

def delta_Et_calculation(df_input):
    df_current = df_input[(df_input['Relaxation'] == 1)].copy()

    Et1 = df_current['Voltage'].iloc[0:5].median()
    Et2 = df_current['Voltage'].iloc[-1:-6:-1].median()

    delta_Et = Et2 - Et1
    return delta_Et

# def delta_V_calculation(df_input):
#     df_pulse = df_input.copy()
#     df_current = df_input[(df_input['Relaxation'] == 1)].copy()

#     current = df_input[(df_input['Relaxation'] == 0)]['Current'].mean()


#     Es1 = df_current['Voltage'].iloc[0]
#     Es2 = df_current['Voltage'].iloc[-1]

#     t1 = df_current['TestTime'].iloc[0]
#     t2 = df_current['TestTime'].iloc[-1]

#     delta_Es = Es2 - Es1
#     delta_t = np.sqrt(t2) - np.sqrt(t1)
#     return -1 / current * delta_Es / delta_t

def tau_calculation(df_pulse):
    df_current = df_pulse[df_pulse['Relaxation'] == 1].iloc[1:]

    tau = df_current['TestTime'].max() - df_current['TestTime'].min()
    return tau


def diffusion_coefficient(delta_Es, delta_Et, tau):
    radius = 5e-6

    D = 4/(9 * np.pi) * radius**2 / tau * (delta_Es / delta_Et)**2
    return D


def global_calculation_ICI(df_nested):
    df_total = pd.DataFrame()
    for pulse, df_pulse in df_nested.items():
        pulse_current = abs(df_pulse[df_pulse['Relaxation'] == 0]['Current']).mean()

        if len(df_pulse[df_pulse['Relaxation'] == 0]) >= 2 and len(df_pulse[df_pulse['Relaxation'] == 1]) >= 2:

            # resistance = ohmic_resistance(df_pulse, pulse_current)
            delta_Es = delta_Es_calculation(df_pulse)
            delta_Et = delta_Et_calculation(df_pulse)
            tau = tau_calculation(df_pulse)
            D = diffusion_coefficient(delta_Es, delta_Et, tau)

            # D = delta_V_calculation(df_pulse)


            df_coefficient = pd.DataFrame({'Pulse': pulse,
                            'Cycle': df_pulse['Cycle'].iloc[0],
                            'TestTime': df_pulse['TestTime'].iloc[0],
                            'SOC': df_pulse['SOC'].iloc[0],
                            'Voltage': df_pulse['Voltage'].iloc[0],
                            'Diffusion Coefficient': D,
                            # 'Resistance': resistance,
                            # 'delta_Es': delta_Es,
                            # 'delta_Et': delta_Et,
                            'tau': tau,
                            'State': df_pulse['State'].iloc[0],},
                            index=['Pulse'])

            df_total = pd.concat([df_total, df_coefficient])
    return df_total