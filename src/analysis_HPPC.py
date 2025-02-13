import pandas as pd
import numpy as np

def pulse_number_HPPC(df_input):
    # A pulse starts the time just before the pulse and ends the time just before the next pulse at the end of the relaxation time
    df = df_input.copy()

    df_input['neg_pulse'] = (df['Current'] <= 0.9 * df['Current'].min()).astype(int).diff().fillna(0).gt(0).cumsum().ffill()
    df_input['pos_pulse'] = (df['Current'] >= 0.9 * df['Current'].max()).astype(int).diff().fillna(0).gt(0).cumsum().ffill()
    df['Pulse'] = np.maximum(df_input['neg_pulse'], df_input['pos_pulse'])

    for pulse in df['Pulse'].unique():
        start_time = df[df['Pulse'] == pulse]['TestTime'].min()
        end_time = df[df['Pulse'] == pulse]['TestTime'].max()

        if not df.loc[(df['TestTime'] <= start_time) & (df['normcurrent'] == 0), 'TestTime'].empty:
            pulse_start = df.loc[(df['TestTime'] <= start_time) & (df['normcurrent'] == 0), 'TestTime'].iloc[-1]
            df.loc[(df['TestTime'] >= pulse_start) & (df['TestTime'] <= end_time), 'Pulse'] = pulse

    for pulse in df['Pulse'].unique():
        df.loc[df['Pulse'] == pulse, 'State'] = df[df['Pulse'] == pulse]['State'].mode()[0]
        df.loc[df['Pulse'] == pulse, 'Cycle'] = df[df['Pulse'] == pulse]['Cycle'].mode()[0]

    df['discharge_pulse'] = (df['Current'] <= 0.9 * df['Current'].min()).astype(int)
    df['charge_pulse'] = (df['Current'] >= 0.9 * df['Current'].max()).astype(int)

    df_nested = {int(pulse): df[df['Pulse'] == pulse] for pulse in df['Pulse'].unique()}

    return df_nested


def internal_resistance(df_pulse, df_charge, df_discharge):
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


def pulse_power_capability(Vmin, Vmax, OCV, R_discharge, R_charge):
    P_discharge = Vmin * (OCV - Vmin) / R_discharge 
    P_charge = Vmax * (Vmax - OCV) / R_charge 
    return P_discharge, P_charge


def global_calculation_HPPC(df_nested):
    df_total = pd.DataFrame()

    df = pd.concat(df_nested.values())
    Vmin = df['Voltage'].min()
    Vmax = df['Voltage'].max()

    for pulse, df_pulse in df_nested.items():
        print('Pulse: '+str(pulse))

        df_charge = df_pulse[df_pulse['charge_pulse'] == 1]
        df_discharge = df_pulse[df_pulse['discharge_pulse'] == 1]

        OCV = df_pulse['Voltage'].iloc[0]

        if len(df_charge) >= 2 and len(df_discharge) >= 2:

            R_discharge, R_charge = internal_resistance(df_pulse, df_charge, df_discharge)
            P_discharge, P_charge = pulse_power_capability(Vmin, Vmax, OCV, R_discharge, R_charge)

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