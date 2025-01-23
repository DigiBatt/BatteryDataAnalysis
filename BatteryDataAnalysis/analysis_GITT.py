import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import numpy as np

def pulse_number(df_input):
    # A pulse starts the time just before the pulse and ends the time just before the next pulse at the end of the relaxation time
    df = df_input.copy()

    df['is_transition'] = (df['Current/mA'] != 0).astype(int).diff().fillna(0).gt(0).cumsum()
    df['Pulse'] = df['is_transition'].where(df['Current/mA'] != 0).ffill().fillna(0).astype(int).shift(-1)
    df['Relaxation'] = (df['Current/mA'] == 0).astype(int)
    
    df.drop(columns=['is_transition'], inplace=True)

    return df


def potential_slope(df_input, tau):
    df_pulse = df_input.copy()
    df_current = df_pulse[(df_pulse['Relaxation'] == 0)]

    t_relax = df_current['TestTime'] - df_current['TestTime'].min()
    time = (t_relax + tau)**0.5 - (t_relax)**0.5
    
    start_time, end_time = time.min(), time.max()
    min_time = 0.05 * (end_time - start_time) + start_time

    voltage_max = df_current.loc[time == start_time, 'Voltage/V'].iloc[0]
    voltage_min = df_current.loc[time <= min_time, 'Voltage/V'].iloc[0]

    slope = (voltage_max - voltage_min) / (min_time - start_time)

    return slope


def ohmic_resistance(df_input):
    df_pulse = df_input.copy()

    pulse_current = abs(df_pulse[df_pulse['Relaxation'] == 0]['Current/mA']).mean() * 1e-3

    # Given that the drop start at 95% of the mean pulse current
    time_end = df_pulse[abs(df_pulse['Current/mA']) >= 0.95 * pulse_current]['TestTime'].min()

    E1 = df_pulse['Voltage/V'].iloc[0]
    E2 = df_pulse.loc[df_pulse['TestTime'] == time_end, 'Voltage/V'].iloc[0]

    resistance = abs((E2 - E1) / pulse_current)

    # print('Resistance: '+str(resistance))

    return resistance

def delta_Es_calculation(df_input):
    df_pulse = df_input.copy()

    Es1 = df_pulse['Voltage/V'].iloc[0]
    Es2 = df_pulse['Voltage/V'].iloc[-1]

    delta_Es = Es2 - Es1
    # print('ΔEs: '+str(delta_Es))

    return delta_Es

def delta_Et_calculation(df_input):
    df_current = df_input[(df_input['Relaxation'] == 0)].copy()

    Et1 = df_current['Voltage/V'].iloc[0]
    Et2 = df_current['Voltage/V'].iloc[-1]

    delta_Et = Et2 - Et1
    # print('ΔEt: '+str(delta_Et))

    return delta_Et

def tau_calculation(df_pulse):
    df_current = df_pulse[df_pulse['Relaxation'] == 0]

    tau = df_current['TestTime'].max() - df_current['TestTime'].min()
    # print('τ: '+str(tau))

    return tau


def diffusion_coefficient(delta_Es, slope, tau):
    # Thomas Schied formula
    radius = 5e-6

    D = 4/(9 * np.pi) * ((radius * delta_Es) / (tau * slope))**2
    # print('D: '+str(D))

    return D

def global_calculation(df):
    df_total = pd.DataFrame()
    results = []
    for pulse in df['Pulse'].unique():

        print('Pulse: '+str(pulse))
        df_pulse = df[df['Pulse'] == pulse].copy()

        resistance = ohmic_resistance(df_pulse)
        delta_Es = delta_Es_calculation(df_pulse)
        delta_Et = delta_Et_calculation(df_pulse)
        tau = tau_calculation(df_pulse)
        slope = potential_slope(df_pulse, tau)
        D = diffusion_coefficient(delta_Es, slope, tau)

        result_dict = {
            'Pulse': pulse,
            'dE/d(√t)': slope,
            'Resistance': resistance,
            'delta_Es': delta_Es,
            'delta_Et': delta_Et,
            'τ': tau,
            'D': D,
        }
        results.append(result_dict)

        df_total = pd.concat([df_total, df_pulse])

    results_df = pd.DataFrame(results)
    print(results_df)

    return df_total, results_df