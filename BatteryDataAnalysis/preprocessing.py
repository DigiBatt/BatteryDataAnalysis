import os
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import process

COLUMN_NAME_MAPPING = {
    'SysTime': ['TestTime', 'Time', 'Time/Sec', 'Time [datetime]', 'SysTime', 'DPt Time', 'TestTime [h]', 't', 'DPtTime'],
    'Voltage': ['Voltage', 'Voltage/V', 'V', 'Volt', "Voltage [V]", 'Volts'],
    'Current': ['Current', 'Current/mA', 'I', 'Current(A)', 'I/mA', "Current [mA]", 'Amps'],
    'Capacity': ['Capacity', 'Capacity/mAh', 'Capacity(Ah)', 'Q', "Capacity [mAh]", 'Amp-hr', 'Amphr'],
    'Cycle': ['Cycle', 'Cycle#', 'Cycle [#]', 'CycleNo', 'Cyc#'],
    'State': ['State', 'Charge', 'C/D', 'Charge/Discharge'],
    'discharging': ['discharging', 'discharge', 'D'],
}

def preprocessing_files(file_path, column_names=None, cycle=None):
    print('File : '+str(os.path.basename(file_path)))

    table = pq.read_table(file_path)
    df = table.to_pandas()

    # df['Cyc'] = 1
    # df['State'] = 'D'
    # df = df.drop(columns='Amphr')
    # # df = df.drop(columns='State')


    print('Length : '+str(len(df)))
    if column_names:
        df = df.rename(columns=column_names)

    print('Column names : '+str(df.columns))
    
    df = standardize_column_names(df)

    print('Column names after standardization : '+str(df.columns))

    if cycle:
        df = df[df['Cycle'].isin(cycle)]

    df = charging_state(df)
    df = process_useful_columns(df)

    df = df.dropna(subset=['Voltage', 'Current', 'Capacity']) 

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
    

def standardize_column_names(df):
    """
    Standardizes column names of the DataFrame using predefined mappings and fuzzy matching
    
    Parameters:
    - df: The input DataFrame with raw column names
    
    Returns:
    - df: DataFrame with standardized column names
    """
    reverse_mapping = {alias: standard_name for standard_name, aliases in COLUMN_NAME_MAPPING.items() for alias in aliases}
    known_standard_columns = list(COLUMN_NAME_MAPPING.keys())
    
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
    
    Parameters:
    - df: The input DataFrame with standardized column names
    
    Returns:
    - df: DataFrame with processed columns
    """
    df = df_input.copy()

    # SysTime to TestTime (seconds)
    if 'SysTime' in df.columns:
        if df['SysTime'].dtype == 'float64':
            df["TestTime"] = df["SysTime"] - df["SysTime"].min()
        
        else:
            df["SysTime"] = pd.to_datetime(df["SysTime"], errors='coerce')
            reference_time = df["SysTime"].min()
            df["TestTime"] = (df["SysTime"] - reference_time).dt.total_seconds()
            df = df.dropna(subset=['TestTime'])

    # Capacity (Ah)
    if 'Capacity' not in df.columns:
        for charging_state in ['C', 'D']:
            for cycle in df['Cycle'].unique():
                print('Cycle '+str(cycle/max(df['Cycle'].unique())))

                df_cycle = df[(df['Cycle'] == cycle) & (df['State'] == charging_state)]
                df.loc[(df['Cycle'] == cycle) & (df['State'] == charging_state), 
                    'Capacity'] = abs((df_cycle['Current'] * df_cycle['TestTime'].diff()).cumsum()) / 3600
        df['Capacity'] = df['Capacity'].ffill()

    # SOC (%)
    df["SOC"] = df.groupby(["Cycle", "State"])["Capacity"].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

    return df


def charging_state(df):
    """
    Standardizes the charging state column based on the current sign and voltage trend, or try fuzzy matching.
    
    Parameters:
    - df (DataFrame): The input DataFrame with standardized column names.
    
    Returns:
    - DataFrame: DataFrame with standardized charging state column.
    """
    # Fuzzy matching to identify the charging state names
    if 'State' in df.columns:
            known_names = {
                'charging': ['C', 'charge', 'charging'],
                'discharging': ['D', 'discharge', 'discharging']
            }
            states = df['State'].unique()
            charge_match = process.extractOne(states, known_names['charging'], score_cutoff=50)
            discharge_match = process.extractOne(states, known_names['discharging'], score_cutoff=50)
        
            if charge_match:
                df.loc[df['State'] == charge_match[0], 'State'] = 'C'
            if discharge_match:
                df.loc[df['State'] == discharge_match[0], 'State'] = 'D'
            return df
    
    # In case a discharging column contains 0 and 1 values
    if 'discharging' in df.columns and df['discharging'].isin([0, 1]).all():
        df.loc[df['discharging'] == 1, 'State'] = 'D'
        df.loc[df['discharging'] != 1, 'State'] = 'C'

        return df
            
    # Link the current sign to the charging state
    if df['Current'].min() * df['Current'].max() < 0:
        if df[df['Current'] < 0]['Voltage'].diff().mean() < 0:
            df.loc[df['Current'] < 0, 'State'] = 'D'
            df.loc[df['Current'] > 0, 'State'] = 'C'
        else:
            df.loc[df['Current'] < 0, 'State'] = 'C'
            df.loc[df['Current'] > 0, 'State'] = 'D'
            df['Current'] = -df['Current']
            
        df['State'] = df['State'].ffill()
        return df

    print('Could not identify charging state')
    return False