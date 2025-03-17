import pandas as pd
import numpy as np
from whittaker_eilers import WhittakerSmoother

def pulse_number_ICI(df_input):
    # A pulse starts the time just before the pulse and ends the time just before the next pulse at the end of the relaxation time
    df = df_input.copy()

    # df['Pulse'] = (df['normcurrent'] == 0).astype(int).diff().fillna(0).gt(0).cumsum().ffill().shift(-1)
    # df['Relaxation'] = (df['normcurrent'] == 0).astype(int)

    # for pulse in df['Pulse'].unique():
    #     start_time = df[df['Pulse'] == pulse]['TestTime'].min()
    #     end_time = df[df['Pulse'] == pulse]['TestTime'].max()

    #     if not df.loc[(df['TestTime'] <= start_time) & (df['normcurrent'] == 0), 'TestTime'].empty:
    #         pulse_start = df.loc[(df['TestTime'] <= start_time) & (df['normcurrent'] == 0), 'TestTime'].iloc[-1]
    #         df.loc[(df['TestTime'] >= pulse_start) & (df['TestTime'] <= end_time), 'Pulse'] = pulse

    # for pulse in df['Pulse'].unique():
    #     df.loc[df['Pulse'] == pulse, 'State'] = df[df['Pulse'] == pulse]['State'].mode()[0]
    #     df.loc[df['Pulse'] == pulse, 'Cycle'] = df[df['Pulse'] == pulse]['Cycle'].mode()[0]

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


def ohmic_resistance(df_input, pulse_current):
    df_pulse = df_input.copy()
    df_relax = df_pulse[(df_pulse['Relaxation'] == 1)].copy()

    E1 = df_pulse['Voltage'].iloc[0]
    E2 = df_relax['Voltage'].iloc[0]

    resistance = abs((E2 - E1) / pulse_current)
    return resistance
    

def delta_Es_calculation(df_input):
    df_pulse = df_input.copy()

    Es1 = df_pulse['Voltage'].iloc[0]
    Es2 = df_pulse['Voltage'].iloc[-1]

    t1 = df_pulse[df_pulse['Relaxation'] == 1]['TestTime'].iloc[-1]
    t2 = df_pulse['TestTime'].iloc[-1]

    delta_Es = Es2 - Es1
    return delta_Es / (t2 - t1)

def delta_Et_calculation(df_input):
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
    cst = 1e-5

    D = 4/np.pi * (delta_Es / delta_Et)**2 * cst**2
    return D

import plotly.express as px
import plotly.graph_objects as go

def global_calculation_ICI(df_nested):
    df_total = pd.DataFrame()
    for pulse, df_pulse in df_nested.items():
        pulse_current = abs(df_pulse[df_pulse['Relaxation'] == 0]['Current']).mean()

        if len(df_pulse[df_pulse['Relaxation'] == 0]) > 5 and len(df_pulse[df_pulse['Relaxation'] == 1]) > 5:

            # if pulse == 454:
            #     df_current = df_pulse[(df_pulse['Relaxation'] == 1)].copy()

            #     fig = px.scatter(df_current, x='TestTime', y='Voltage', color='State', title=pulse)
            #     fig.show()

            #     signal_noise = np.diff(df_current['Voltage'], 2)
            #     signal_noise_absolute = np.abs(signal_noise).reshape(-1, 1)
            #     signal_noise_normalized = signal_noise_absolute / abs(df_current['Voltage']).max()
            #     weights = np.exp(-signal_noise_normalized)
            #     weights_padded = np.concatenate(([weights[0]], weights, [weights[-1]]))

            #     whittaker_smoother = WhittakerSmoother(lmbda=1e-2, order=1, data_length=len(df_current), x_input=df_current['TestTime'], weights=weights_padded)
            #     df_current['Voltage'] = whittaker_smoother.smooth(df_current['Voltage'].values)

            #     df = df_current
            #     fig = px.scatter(df, x='TestTime', y='Voltage', color='State', title=pulse)
            #     fig.show()

            #     fig = go.Figure()
            #     fig.add_trace(go.Scatter(x=np.sqrt(df['TestTime'] - df['TestTime'].iloc[0]), y=df['Voltage']))
            #     fig.show()

            #     fig = go.Figure()
            #     fig.add_trace(go.Scatter(x=np.sqrt(df['TestTime'] - df['TestTime'].iloc[0]), y=df['Voltage'].diff() / np.sqrt(df['TestTime'] - df['TestTime'].iloc[0]).diff()))
            #     fig.show()
                
            # print(pulse)
            resistance = ohmic_resistance(df_pulse, pulse_current)
            delta_Es = delta_Es_calculation(df_pulse)
            delta_Et = delta_Et_calculation(df_pulse)
            # print(delta_Es, delta_Et)
            tau = tau_calculation(df_pulse)
            D = diffusion_coefficient(delta_Es, delta_Et, tau)

            # D = delta_V_calculation(df_pulse)


            df_coefficient = pd.DataFrame({'Pulse': pulse,
                            'Cycle': df_pulse['Cycle'].iloc[0],
                            'TestTime': df_pulse['TestTime'].iloc[0],
                            'SOC': df_pulse['SOC'].iloc[0],
                            'Voltage': df_pulse['Voltage'].iloc[0],
                            'Diffusion Coefficient': D,
                            'Resistance': resistance,
                            # 'delta_Es': delta_Es,
                            # 'delta_Et': delta_Et,
                            'tau': tau,
                            'State': df_pulse['State'].iloc[0],},
                            index=['Pulse'])

            df_total = pd.concat([df_total, df_coefficient])
    return df_total