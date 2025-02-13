import os
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import process
import numpy as np

COLUMN_NAME_MAPPING = {
    'SysTime': ['TestTime', 'Time', 'Time/Sec', 'Time [datetime]', 'SysTime', 'DPt Time', 'TestTime [h]', 't', 'DPtTime'],
    'Voltage': ['Voltage', 'Voltage/V', 'V', 'Volt', "Voltage [V]", 'Volts'],
    'Current': ['Current', 'Current/mA', 'I', 'Current(A)', 'I/mA', "Current [mA]", 'Amps'],
    'Capacity': ['Capacity', 'Capacity/mAh', 'Capacity(Ah)', 'Q', "Capacity [mAh]", 'Amp-hr', 'Amphr'],
    'Cycle': ['Cycle', 'Cycle#', 'Cycle [#]', 'CycleNo', 'Cyc#'],
    'State': ['State', 'Charge', 'C/D', 'Charge/Discharge'],
    # 'discharging': ['discharging', 'discharge', 'D'],
}
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
    print('File : '+str(os.path.basename(file_path)))
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.parquet':
        table = pq.read_table(file_path)
        df = table.to_pandas()
    elif file_ext == '.csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Format de fichier non supporté : {file_ext}")
    
    if debug_func is not None:
        df = debug_func(df)

    print('Length : '+str(len(df)))
    if column_names:
        df = df.rename(columns=column_names)

    df = standardize_column_names(df, column_names)

    for col in df.columns:
        if col not in ['Voltage', 'Current', 'SysTime']:
            df = df.drop(col, axis=1)

    df = df.groupby(["SysTime"]).median().reset_index()

    if cycle:
        df = df[df['Cycle'].isin(cycle)]

    df = df.dropna(subset=['Voltage', 'Current'])
    df = process_useful_columns(df)
    df = df.dropna(subset=['Capacity'])

    return df

def fuzzy_match_column(column, known_columns):
    """
    Performs fuzzy matching to guess the most appropriate column name.
    
    Parameters:
    - column: The raw column name to be matched.
    - known_columns: List of known standardized column names.
    
    Returns:
    - str: The best guess for the column name, or the original column if no good match is found.
    """
    # Fuzzy match the column name against the known columns
    best_match, score, _ = process.extractOne(column, known_columns)
    
    # If the score is above a threshold (e.g., 70), return the match; otherwise, return the original column name
    if score > 70:
        return best_match
    else:
        return column
    

def standardize_column_names(df, column_names):
    """
    Standardizes column names of the DataFrame using predefined mappings and fuzzy matching
    
    Parameters:
    - df: The input DataFrame with raw column names
    
    Returns:
    - df: DataFrame with standardized column names
    """
    reverse_mapping = {alias: standard_name for standard_name, aliases in COLUMN_NAME_MAPPING.items() for alias in aliases}
    known_standard_columns = list(COLUMN_NAME_MAPPING.keys())

    if column_names:
        known_standard_columns = [col for col in known_standard_columns if col not in column_names.values()]
    
    # Rename columns based on predefined mappings or fuzzy matching
    new_columns = []
    for col in df.columns:
        if col in reverse_mapping:
            new_columns.append(reverse_mapping[col])
        else:
            new_columns.append(fuzzy_match_column(col, known_standard_columns))
    
    df.columns = new_columns
    df = df.loc[:, ~df.columns.duplicated()]
    
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
    if 'SysTime' in df.columns:
        if df['SysTime'].dtype == 'float64':
            df["TestTime"] = df["SysTime"] - df["SysTime"].min()
        
        else:
            df["SysTime"] = pd.to_datetime(df["SysTime"], errors='coerce')
            reference_time = df["SysTime"].min()
            df["TestTime"] = (df["SysTime"] - reference_time).dt.total_seconds()
            df = df.dropna(subset=['TestTime'])

    ## Current (A)
    if df['Current'].max() > 15:
        df['Current'] = df['Current'] / 1000
    
    ## Capacity (Ah)
    df['Capacity'] = (df['Current'] * df['TestTime'].diff()).cumsum() / 3600
    df['Capacity'] = df['Capacity'] - df['Capacity'].min()

    ## Cycle and State
    # Rounding of the current near 0
    df_nocurrent = df[abs(df['Current']) <= abs(df['Current'].max()) * 0.05]
    nocurrent_max = abs(df_nocurrent['Current']).max()

    df['normcurrent'] = df['Current']
    if nocurrent_max != 0 and not np.isnan(nocurrent_max):
        df['normcurrent'] = round(df['normcurrent'] / (nocurrent_max * 2.1)) * (nocurrent_max * 2.1)

    # Local charging/discharging state and counting each state
    df.loc[df['normcurrent'] < 0, 'Local_state'] = 'D'
    df.loc[df['normcurrent'] > 0, 'Local_state'] = 'C'
    df['Local_state'] = df['Local_state'].ffill().infer_objects(copy=False)

    df['Group'] = (df['Local_state'] != df['Local_state'].shift()).cumsum()
    df_group = df.groupby('Group')['TestTime'].agg(lambda x: x.max() - x.min() + 1)

    # Global charging/discharging state according to the duration of each local state
    for group in df['Group'].unique():
        if df_group.loc[group] > 200:
            df.loc[df['Group'] == group, 'State'] = ('C' if df[df['Group'] == group]['normcurrent'].mean() >= 0 
                                                     else 'D')
    df['State'] = df['State'].ffill()

    # First cycle starts with the first discharge and ends with the end of the next charge
    fist_discharge_time = df[df['State'] == 'D']['TestTime'].min()
    df_discharge = df[df['TestTime'] > fist_discharge_time]

    df['Cycle'] = 0
    df.loc[df['TestTime'] > fist_discharge_time, 'Cycle'] = (((df_discharge['State'] != df_discharge['State'].shift()
                                                               ).cumsum() + 1) // 2
                                                               ).astype(int)

    ## State of charge for each cycle
    df["SOC"] = df.groupby(["Cycle"])["Capacity"].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

    return df