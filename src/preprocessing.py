import os
import time
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import process
from sklearn.cluster import KMeans
import numpy as np
import chardet
import json

def read_file(file_path, column_names=None, cycle=None, debug_func=None, input='path'):
    """Reads the file and returns a list of DataFrames
    
    Parameters
    ----------
    file_path : str
        Path to the file to process
    
    Returns
    -------
    list
        List of DataFrames containing the data of the file
    """
    if input == 'dataframe':
        df = preprocessing_files(file_path, column_names, cycle, debug_func)
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataframe.csv')
        return [df], [file_path]

    else:
        print('File : '+str(os.path.basename(file_path)))
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == '.parquet':
            df = pq.read_table(file_path).to_pandas()
            df = preprocessing_files(df, column_names, cycle, debug_func)
            return [df], [file_path]
        
        elif file_ext == '.csv':
            encoding = detect_encoding(file_path)
            sep = detect_separator(file_path, encoding)
            skip_rows = find_header_row(file_path, encoding, sep)

            df = pd.read_csv(file_path, encoding=encoding, sep=sep, skiprows=skip_rows)
            df = preprocessing_files(df, column_names, cycle, debug_func)
            return [df], [file_path]
        
        elif file_ext == '.txt':
            encoding = detect_encoding(file_path)
            sep = detect_separator(file_path, encoding)
            skip_rows = find_header_row(file_path, encoding, sep)

            df = pd.read_csv(file_path, encoding=encoding, sep=sep, skiprows=skip_rows, index_col=False)
            df = preprocessing_files(df, column_names, cycle, debug_func)
            return [df], [file_path]
        
        elif file_ext in ['.xlsx', '.xls']:
            engine = detect_excel_engine(file_path)
            sheets_dict = pd.read_excel(file_path, sheet_name=None, engine=engine)

            df_list, file_path_list = [], []
            for sheet_name in sheets_dict.keys():
                skip_rows = find_excel_header(file_path, engine, sheet_name)
                df = pd.read_excel(file_path, sheet_name=sheet_name, engine=engine, skiprows=skip_rows)

                try:
                    print(f'\nSheet {sheet_name}')
                    df_list.append(preprocessing_files(df, column_names, cycle, debug_func))
                    new_file_path = add_suffix_to_filename(file_path, sheet_name)
                    file_path_list.append(new_file_path)
                except Exception as e:
                    print(f'Error preprocessing sheet: {e}') 
            return df_list, file_path_list
        else:
            raise ValueError(f"Unsupported file format : {file_ext}")


def preprocessing_files(df, column_names=None, cycle=None, debug_func=None):
    """Preprocessing the file to format it for future analysis

    It reads the file and applies the preprocessing functions of renaming and calculating the useful columns:
    TestTime, normcurrent, State, Cycle, Capacity, SOC and C-Rate

    Parameters
    -------
    file_path : str
        Path to the file to process
    column_names : Dict, optionnal
        Dictionary containing the column names to be used for the analysis
    cycle : int, optionnal
        List of cycle numbers to process
    debug_func : function, optionnal
        Function to apply to the DataFrame before preprocessing to remove specific bugs of one dataset

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the preprocessed data
    """
    start_time = time.time()

    if debug_func is not None:
        df = debug_func(df)

    print('Length : '+str(len(df)))

    df = standardize_column_names(df, column_names)
    for col in ['SysTime', 'Voltage', 'Current']:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")

    df = process_useful_columns(df)

    if cycle:
        df = df[df['Cycle'].isin(cycle)]

    end_time = time.time()
    print('Preprocessing Time : '+str(int(end_time - start_time))+'s')
    return df


def standardize_column_names(df_input, column_names):
    """Standardizes column names of the DataFrame using predefined mappings and fuzzy matching
    
    Parameters
    -------
    df : pandas.DataFrame
        The input DataFrame with raw column names
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with only the three columns: Voltage, Current and SysTime
    """
    df = df_input.copy()

    COLUMN_NAME_MAPPING = {
        'SysTime': ['TestTime', 'Time', 'Time/Sec', 'Time [datetime]', 'SysTime', 'DPt Time', 'TestTime [h]', 't', 
                    'DPtTime', 'time/s', 'test_time_millisecond', 'DPt-Time', 'Duration (sec)', 'Date_Time'],
        'Voltage': ['Voltage', 'Voltage/V', 'V', 'Volt', "Voltage [V]", 'Volts', 'voltage_volt', 'mV', 'V', 'Voltage(V)'],
        'Current': ['Current', 'Current/mA', 'I', 'Current(A)', 'I/mA', "Current [mA]", 'Amps', 'current_ampere', 'mA', 'A'],
            }
    
    if column_names:
        df = df.rename(columns=column_names)

    for standard_col in ['Voltage', 'Current', 'SysTime']:
        if standard_col not in df.columns:
            for col in df.columns:
                if col in COLUMN_NAME_MAPPING[standard_col] and standard_col not in df.columns:
                    df = convert_unit(df, col, standard_col)
                    df = df.rename(columns={col: standard_col})

            for col in df.columns:
                if standard_col not in df.columns:
                    best_match, score, _ = process.extractOne(col, COLUMN_NAME_MAPPING[standard_col])
                    if score > 90:
                        df = convert_unit(df, col, standard_col)
                        df = df.rename(columns={col: standard_col})

    for col in df.columns:
        if col not in ['Voltage', 'Current', 'SysTime']:
            df = df.drop(col, axis=1)
        else:
            df = df.dropna(subset=[col])

    return df


def convert_unit(df_input, col, standard_col):
    """Converts the unit of the column to the correct unit

    Works only if the unit is in the column name
    
    Parameters
    -------
    df : pandas.DataFrame
        The input DataFrame with raw column names
    col : str
        The column name
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with the unit converted
    """
    df = df_input.copy()
    if standard_col == 'Current':
        if 'mA' in col:
            df[col] = df[col] / 1000
    elif standard_col == 'Voltage':
        if 'mV' in col:
            df[col] = df[col] / 1000
    elif standard_col == 'SysTime':
        if 'ms' in col or 'millisecond' in col:
            df[col] = df[col] / 1000
    return df


def process_useful_columns(df_input):
    """Processes useful columns for future analysis

    The processed columns are: TestTime, Capacity, Cycle, State and SOC
    
    Parameters
    -------
    df : pandas.DataFrame
        The input DataFrame with standardized column names
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with processed columns
    """
    df = df_input.copy()

    ## SysTime to TestTime (seconds)
    if df['SysTime'].dtype == 'float64' or df['SysTime'].dtype == 'int64':
        df["TestTime"] = df["SysTime"] - df["SysTime"].min()
    else:
        df["SysTime"] = pd.to_datetime(df["SysTime"], errors='coerce')
        df["TestTime"] = (df["SysTime"] - df["SysTime"].min()).dt.total_seconds()
        df = df.dropna(subset=['TestTime'])

    # df["TestTime"] = df["TestTime"].astype(int)
    df = df.groupby(["TestTime"]).median().reset_index()

    ## Cycle and State
    # Clustering of the current repartition
    df.loc[df.index[0], 'Current'] = 0
    X = df['Current'].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=8, random_state=0)
    df['Cluster'] = kmeans.fit_predict(X)

    # Rounding to 0 the cluster closest to 0
    zero_cluster_idx = np.argmin(np.abs(kmeans.cluster_centers_.flatten()))
    df['normcurrent'] = df.apply(lambda row: 0 if row['Cluster'] == zero_cluster_idx else row['Current'], axis=1)

    # Local charging/discharging state and counting each state
    df.loc[df['normcurrent'] < 0, 'Local_state'] = 'D'
    df.loc[df['normcurrent'] > 0, 'Local_state'] = 'C'
    df['Local_state'] = df['Local_state'].ffill().infer_objects(copy=False).fillna('R')

    df['Group'] = (df['Local_state'] != df['Local_state'].shift()).cumsum()
    df_groupby_testduration = df.groupby('Group')['TestTime'].agg(lambda x: x.max() - x.min())

    # Global charging/discharging state according to the duration of each local state
    threshold_pulse_duration = 200
    successive_pulse_count, successive_pulse_start, successive_pulse_end = 0, 0, 0
    for group in df['Group'].unique():
        df_group = df[df['Group'] == group]
        df_group_current = df_group[df_group['normcurrent'] != 0]
        full_cycling = (df_group['Voltage'].max() >= 0.9*df['Voltage'].max() and df_group['Voltage'].min() <= 1.1*df['Voltage'].min())

        # if a same local state is maintained for at least the thrsehold duration, the state corresponds to the voltage difference between the start and the end of the group
        if (df_groupby_testduration.loc[group] > threshold_pulse_duration) and (len(df_group_current) != 0):
            df.loc[df['Group'] == group, 'State'] = ('C' if df_group_current['Voltage'].iloc[-1] - df_group_current['Voltage'].iloc[0] >= 0
                                                     else 'D')
            # if the last groups were pulses: after the threshold duration, it keeps the global state between the start and the end of the pulses
            if successive_pulse_end - successive_pulse_start > 200:
                df.loc[(df['TestTime'] >= successive_pulse_start) & (df['TestTime'] <= successive_pulse_end), 
                        'State'] = ('C' if successive_voltage_end - successive_voltage_start >= 0 
                                    else 'D')
            successive_pulse_count = 0

        elif len(df_group_current) != 0:
            # if it is not a fast cycling, it keeps the start and the end voltages of the successive pulses
            if not full_cycling:
                if successive_pulse_count == 0:
                    successive_pulse_start = df_group['TestTime'].iloc[0]
                    successive_voltage_start = df_group_current['Voltage'].iloc[0]
                successive_pulse_end = df_group['TestTime'].iloc[-1]
                successive_voltage_end = df_group_current['Voltage'].iloc[-1]
                successive_pulse_count += 1
            # if it is a fast cycling (cycling that lasts less than the threshold duration), the state corresponds to the voltage difference between the start and the end of the group
            else:
                df.loc[df['Group'] == group, 'State'] = ('C' if df_group['Voltage'].iloc[-1] - df_group['Voltage'].iloc[0] >= 0
                                                         else 'D')
        else:
            df.loc[df['Group'] == group, 'State'] = ('C' if df_group['Voltage'].iloc[-1] - df_group['Voltage'].iloc[0] >= 0
                                                     else 'D')
    # if the last groups of the dataframes are successive pulses
    if successive_pulse_count > 0:
        df.loc[(df['TestTime'] >= successive_pulse_start) & (df['TestTime'] <= successive_pulse_end), 
               'State'] = ('C' if successive_voltage_end - successive_voltage_start >= 0 else 'D')
    df['State'] = df['State'].ffill()

    # First cycle starts with the first discharge and ends with the end of the next charge    
    fist_discharge_time = df[df['State'] == 'D']['TestTime'].min()
    df_discharge = df[df['TestTime'] > fist_discharge_time]
    df['Cycle'] = 0
    df.loc[df['TestTime'] > fist_discharge_time, 'Cycle'] = (((df_discharge['State'] != df_discharge['State'].shift()
                                                               ).cumsum() + 1) // 2
                                                               ).astype(int)
    df['Cycle'] = df['Cycle'].astype(int)

    ## Capacity (Ah)
    if df['Cycle'].nunique() == 1:
        df['Capacity'] = (df['Current'] * df['TestTime'].diff()).cumsum() / 3600
        df['Capacity'] = df['Capacity'].transform(lambda x: x - x.min()).ffill().shift(-1)
        df.dropna(subset=['Capacity'])
    else:
        df['Capacity'] = df.groupby('Cycle').apply(lambda group: (group['Current'] * group['TestTime'].diff()).cumsum() / 3600).reset_index(level=0, drop=True)
        df['Capacity'] = df.groupby('Cycle')['Capacity'].transform(lambda x: x - x.min()).ffill().shift(-1)
        df.dropna(subset=['Capacity'])

    ## State of charge
    df["SOC"] = df.groupby(["Cycle", 'State'])["Capacity"].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

    ## C-Rate
    for cycle in df['Cycle'].unique():
        c_rate = 0
        df_cycle = df[(df['Cycle'] == cycle)].copy()
        df_cycle['Pulse'] = (df_cycle['normcurrent'] != 0).diff().gt(0).cumsum().ffill()
        for pulse in df_cycle['Pulse'].unique():
            df_pulse = df_cycle[(df_cycle['Pulse'] == pulse) & (df_cycle['normcurrent'] != 0)]
            if not df_pulse.empty:
                c_rate += (df_pulse['TestTime'].max() - df_pulse['TestTime'].min()) / 3600
        if c_rate >= 1:
            c_rate = int(c_rate)
        elif c_rate < 1 and c_rate > 0:
            c_rate = 1 / int(1 / c_rate)
        else:
            c_rate = np.nan
        df.loc[(df['Cycle'] == cycle), 'C_Rate'] = c_rate 

    return df
    
def detect_encoding(file_path, num_bytes=10000):
    """Detects the encoding of the file
    
    Parameters
    -------
    file_path : str
    
    Returns
    -------
    str
        Encoding of the file
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read(num_bytes)
    return chardet.detect(raw_data)['encoding']


def detect_separator(file_path, encoding):
    """Detects the separator for the file
    
    Parameters
    -------
    file_path : str
    encoding : str
    
    Returns
    -------
    str
        Separator of the file
    """
    potential_separators = [',', ';', '\t', '|']
    with open(file_path, 'r', encoding=encoding) as f:
        first_lines = [f.readline() for _ in range(5)]
    
    best_sep = ','
    max_count = 0
    for sep in potential_separators:
        count = sum(line.count(sep) for line in first_lines)
        if count > max_count:
            max_count = count
            best_sep = sep
    return best_sep


def find_header_row(file_path, encoding, sep, max_rows=15):
    """Finds the header row of the csv file

    Parameters
    -------
    file_path : str
    encoding : str
    sep : str
    max_rows : int, optional
    
    Returns
    -------
    str
        Header row of the file
    """
    for i in range(max_rows):
        try:
            df_test = pd.read_csv(file_path, encoding=encoding, sep=sep, skiprows=i, nrows=15)
            if all(isinstance(col, str) for col in df_test.columns):
                if not any("Unnamed" in col for col in df_test.columns):
                    return i
        except:
            continue
    return 0

def find_excel_header(file_path, engine, sheet_name, max_rows=15):
    """Finds the header row of a sheet of the excel file
    
    Parameters
    -------
    file_path : str
    engine : str
    sheet_name : str
    max_rows : int, optional
    
    Returns
    -------
    str
        Header row of the file
    """
    for i in range(max_rows):
        df_test = pd.read_excel(file_path, engine=engine, sheet_name=sheet_name, skiprows=i, nrows=15)
        if all(isinstance(col, str) for col in df_test.columns):
            if not any("Unnamed" in col for col in df_test.columns):
                return i
    return 0

def detect_excel_engine(file_path):
    """Detects the engine to use for reading the excel file

    Parameters
    -------
    file_path : str
    
    Returns
    -------
    str
        Engine to use for reading the excel file
    """
    try:
        with open(file_path, 'rb') as f:
            signature = f.read(4)
        if signature == b'PK\x03\x04':  # .xlsx files
            return 'openpyxl'
        else:   # .xls files
            return 'xlrd'
    except:
        return 'openpyxl'

def add_suffix_to_filename(file_path, sheet_name):
    """Add the sheet name to the corresponding folder name

    Parameters
    -------
    file_path : str
    sheet_name : str
    
    Returns
    -------
    str
        New file path with the sheet name added to the filename
    """
    directory, filename = os.path.split(file_path)
    name, ext = os.path.splitext(filename)
    new_filename = f"{name} - sheet {sheet_name}{ext}"
    return os.path.join(directory, new_filename)