import os
import time
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import process
import numpy as np
from whittaker_eilers import WhittakerSmoother

import plotly.express as px

def preprocessing_files(file_path, column_names=None, cycle=None, debug_func=None):
    '''
    Preprocessing of the file
    It reads the file and applies the preprocessing functions of renaming and calculating useful columns for future analysis

    Parameters:
    - file_path: Path to the file to process
    - column_names (optional): Dictionary containing the column names to be used for the analysis
    - cycle (optional): List of cycle numbers to process
    - debug_func (optional): Function to apply to the DataFrame before preprocessing to remove specific bugs of one dataset

    Returns:
    - df: DataFrame containing the preprocessed data
    '''
    start_time = time.time()

    print('File : '+str(os.path.basename(file_path)))
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.parquet':
        df = pq.read_table(file_path).to_pandas()
    elif file_ext == '.csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Format de fichier non supporté : {file_ext}")
    
    if debug_func is not None:
        df = debug_func(df)

    print('Length : '+str(len(df)))

    df = standardize_column_names(df, column_names)
    df = process_useful_columns(df)

    if cycle:
        df = df[df['Cycle'].isin(cycle)]

    end_time = time.time()
    print('Preprocessing Time : '+str(end_time - start_time))
    return df
    

def standardize_column_names(df_input, column_names):
    """
    Standardizes column names of the DataFrame using predefined mappings and fuzzy matching
    
    Parameters:
    - df: The input DataFrame with raw column names
    
    Returns:
    - df: DataFrame with only the three columns: Voltage, Current and SysTime
    """
    df = df_input.copy()

    COLUMN_NAME_MAPPING = {
        'SysTime': ['TestTime', 'Time', 'Time/Sec', 'Time [datetime]', 'SysTime', 'DPt Time', 'TestTime [h]', 't', 'DPtTime', 'time/s'],
        'Voltage': ['Voltage', 'Voltage/V', 'V', 'Volt', "Voltage [V]", 'Volts'],
        'Current': ['Current', 'Current/mA', 'I', 'Current(A)', 'I/mA', "Current [mA]", 'Amps',],
            }
    
    if column_names:
        df = df.rename(columns=column_names)

    for standard_col in ['Voltage', 'Current', 'SysTime']:
        if standard_col not in df.columns:
            for col in df.columns:
                if col in COLUMN_NAME_MAPPING[standard_col] and standard_col not in df.columns:
                    df = df.rename(columns={col: standard_col})

            for col in df.columns:
                if standard_col not in df.columns:
                    best_match, score, _ = process.extractOne(col, COLUMN_NAME_MAPPING[standard_col])
                    if score > 95:
                        df = df.rename(columns={col: standard_col})

    for col in df.columns:
        if col not in ['Voltage', 'Current', 'SysTime']:
            df = df.drop(col, axis=1)
        else:
            df = df.dropna(subset=[col])

    return df


def process_useful_columns(df_input):
    """
    Process useful columns in the DataFrame
    The processed columns are: TestTime, Capacity, Cycle, State and SOC
    
    Parameters:
    - df: The input DataFrame with standardized column names
    
    Returns:
    - df: DataFrame with processed columns
    """
    df = df_input.copy()

    ## SysTime to TestTime (seconds)
    if df['SysTime'].dtype == 'float64':
        df["TestTime"] = df["SysTime"] - df["SysTime"].min()
    else:
        df["SysTime"] = pd.to_datetime(df["SysTime"], errors='coerce')
        df["TestTime"] = (df["SysTime"] - df["SysTime"].min()).dt.total_seconds()
        df = df.dropna(subset=['TestTime'])

    df = df.groupby(["TestTime"]).median().reset_index()


    ## Current (A)
    # Not precise enough
    if df['Current'].max() > 20:
        df['Current'] = df['Current'] / 1000


    ## Cycle and State
    # Rounding of the current near 0
    df['normcurrent'] = df['Current']

    # signal_noise = np.diff(df['Current'], 2)
    # signal_noise_absolute = np.abs(signal_noise).reshape(-1, 1)
    # signal_noise_normalized = signal_noise_absolute / abs(df['Current']).max()
    # weights = np.exp(-signal_noise_normalized)
    # weights_padded = np.concatenate(([weights[0]], weights, [weights[-1]]))

    # whittaker_smoother = WhittakerSmoother(lmbda=1e-1, order=1, data_length=len(df), x_input=df['TestTime'], weights=weights_padded)
    # df['normcurrent'] = whittaker_smoother.smooth(df['Current'].values)


    df_nocurrent = df[abs(df['Current']) <= abs(df['Current']).max() * 0.05]
    nocurrent_max = abs(df_nocurrent['Current']).max()
    
    if nocurrent_max != 0 and not np.isnan(nocurrent_max):
        df.loc[abs(df['Current']) <= nocurrent_max, 'normcurrent'] = round(df['Current'] / (nocurrent_max * 2.1)) * (nocurrent_max * 2.1)

    # Local charging/discharging state and counting each state
    df.loc[df['normcurrent'] < 0, 'Local_state'] = 'D'
    df.loc[df['normcurrent'] > 0, 'Local_state'] = 'C'
    # df.loc[(df['normcurrent'] == 0) & (df['Voltage'].diff() < 0), 'Local_state'] = 'D'
    # df.loc[(df['normcurrent'] == 0) & (df['Voltage'].diff() > 0), 'Local_state'] = 'C'
    df['Local_state'] = df['Local_state'].ffill().infer_objects(copy=False)

    df['Group'] = (df['Local_state'] != df['Local_state'].shift()).cumsum()
    df_group = df.groupby('Group')['TestTime'].agg(lambda x: x.max() - x.min())

    # Global charging/discharging state according to the duration of each local state
    threshold_pulse_duration = 200
    for group in df['Group'].unique():
        if df_group.loc[group] > threshold_pulse_duration:
            df.loc[df['Group'] == group, 'State'] = ('C' if df[df['Group'] == group]['normcurrent'].mean() >= 0 
                                                     else 'D')
    df['State'] = df['State'].ffill()

    # First cycle starts with the first discharge and ends with the end of the next charge
    df_stated = df[df['State'].isin(['C', 'D'])]
    df.loc[df['State'].isin(['C', 'D']), 'Cycle'] = (((df_stated['State'] != df_stated['State'].shift()
                                                               ).cumsum() + 1) // 2
                                                               ).astype(int)
    
    # fist_discharge_time = df[df['State'] == 'D']['TestTime'].min()
    # first_charge_time = df[df['State'] == 'C']['TestTime'].min()
    # df_discharge = df[df['TestTime'] > min(fist_discharge_time, first_charge_time)]
    # df['Cycle'] = 0
    # df.loc[df['TestTime'] > fist_discharge_time, 'Cycle'] = (((df_discharge['State'] != df_discharge['State'].shift()
    #                                                            ).cumsum() + 1) // 2
    #                                                            ).astype(int)

    ## Capacity (Ah)
    df['Capacity'] = df.groupby('Cycle').apply(lambda group: (group['Current'] * group['TestTime'].diff()).cumsum() / 3600).reset_index(level=0, drop=True)
    df['Capacity'] = df.groupby('Cycle')['Capacity'].transform(lambda x: x - x.min()).ffill()
    # df.dropna(subset=['Capacity'])

    ## State of charge
    df["SOC"] = df.groupby(["Cycle", 'State'])["Capacity"].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

    ## C-Rate
    for cycle in df['Cycle'].unique():
        for state in df['State'].unique():
            c_rate = 0
            df_state = df[(df['Cycle'] == cycle) & (df['State'] == state)].copy()
            df_state['Pulse'] = (df_state['normcurrent'] != 0).diff().gt(0).cumsum().ffill()
            for pulse in df_state['Pulse'].unique():
                df_pulse = df_state[(df_state['Pulse'] == pulse) & (df_state['normcurrent'] != 0)]
                if not df_pulse.empty:
                    c_rate += (df_pulse['TestTime'].max() - df_pulse['TestTime'].min()) / 3600
            if c_rate >= 1:
                c_rate = int(c_rate)
            elif c_rate < 1 and c_rate > 0:
                c_rate = 1 / int(1 / c_rate)
            else:
                c_rate = np.nan
            df.loc[(df['Cycle'] == cycle) & (df['State'] == state), 'C_Rate'] = c_rate

    ## 
    for col in ['State', 'Cycle']:
        df[col] = df[col].shift(-1)
        df = df.dropna(subset=[col])

    df['Cycle'] = df['Cycle'].astype(int)

    return df


# df['State'] = df['State'].ffill()#.shift(-1)

# # First cycle starts with the first discharge and ends with the end of the next charge
# # fist_discharge_time = df[df['State'] == 'D']['TestTime'].min()
# # first_charge_time = df[df['State'] == 'C']['TestTime'].min()
# # df_discharge = df[df['TestTime'] > min(fist_discharge_time, first_charge_time)]
# df_stated = df[df['State'].isin(['C', 'D'])]

# # df['Cycle'] = 0
# df.loc[df['State'].isin(['C', 'D']), 'Cycle'] = (((df_stated['State'] != df_stated['State'].shift()
#                                                     ).cumsum() + 1) // 2)
# df['Cycle'] = df['Cycle'].fillna(0).astype(int)


# ## Capacity (Ah)
# df['Capacity'] = df.groupby('Cycle').apply(lambda group: (group['Current'] * group['TestTime'].diff()).cumsum() / 3600).reset_index(level=0, drop=True)
# # df['Capacity'] = df.groupby('Cycle')['Capacity'].transform(lambda x: x - x.min()).ffill()
# df['Capacity'] = df['Capacity'] - df['Capacity'].min()
# df.dropna(subset=['Capacity'])

# ## State of charge
# df["SOC"] = df.groupby(['Cycle', 'State'])["Capacity"].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

# ## C-Rate
# for cycle in df['Cycle'].unique():
#     for state in df['State'].unique():
#         c_rate = 0
#         df_state = df[(df['Cycle'] == cycle) & (df['State'] == state)].copy()
#         df_state['Pulse'] = (df_state['normcurrent'] != 0).diff().gt(0).cumsum().ffill()
#         for pulse in df_state['Pulse'].unique():
#             df_pulse = df_state[(df_state['Pulse'] == pulse) & (df_state['normcurrent'] != 0)]
#             if not df_pulse.empty:
#                 c_rate += (df_pulse['TestTime'].max() - df_pulse['TestTime'].min()) / 3600
#         if c_rate >= 1:
#             c_rate = int(c_rate)
#         elif c_rate < 1 and c_rate > 0:
#             c_rate = 1 / int(1 / c_rate)
#         else:
#             c_rate = np.nan
#         df.loc[(df['Cycle'] == cycle) & (df['State'] == state), 'C-Rate'] = c_rate

# ## 
# for col in ['State', 'Cycle', 'C-Rate']:
#     df[col] = df[col].shift(-1)
#     df = df.dropna(subset=[col])

# df['Cycle'] = df['Cycle'].astype(int)
# df['C-Rate'] = df['C-Rate'].astype(int)