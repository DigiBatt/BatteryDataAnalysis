from preprocessing import preprocessing_files
from analysis_dqdv import calculate_dqdv_for_all_cycle
from analysis_GITT import pulse_number_GITT, global_calculation_GITT
from analysis_HPPC import pulse_number_HPPC, global_calculation_HPPC
from analysis_ICI import pulse_number_ICI, global_calculation_ICI
from plotting import *

import time
import os
import pandas as pd

def process_dqdv(df, 
                 file_path,
                 curve=True, 
                 heatmap=False, 
                 pocv=True, 
                 save=False, 
                 smoothing=True,
                 png=False):
    """
    Process the dQ/dV data for a given file in parquet format
    Plot the dQ/dV curves and the dQ/dV heatmap for every cycles
    
    Parameters:
    - df: DataFrame containing the CCCV data to process
    - file_path: Path to the file to process
    - curve (optional): Boolean indicating whether to plot the dQ/dV curves
    - heatmap (optional): Boolean indicating whether to plot the dQ/dV heatmap
    - pocv (optional): Boolean indicating whether to plot the Voltage over Capacity curves
    - save (optional): Boolean indicating whether to save the plots
    - smoothing (optional): Boolean indicating whether to apply smoothing to the dQ/dV curves

    Returns:
    - df_dqdv: DataFrame containing the dQ/dV data
    """
    start_time = time.time()

    df_dqdv = calculate_dqdv_for_all_cycle(df, smoothing=smoothing)

    fig = plot_test_over_time(df, file_path, png=png)
    # fig.show()

    if curve and df_dqdv is not False:
        fig_dqdv = plot_DQDV_result(df_dqdv, file_path, save, png=png)
        # fig_dqdv.show()

    if heatmap and df_dqdv is not False:
        heatmap_dqdv = plot_dqdv_heatmap(df_dqdv, file_path, save, png=png)
        # heatmap_dqdv.show()

    if pocv and df_dqdv is not False:
        fig_pocv = plot_pocv(df, file_path, save, png=png)
        # fig_pocv.show()

    print('dQ/dV Time : '+str(int(time.time() - start_time))+' s')
    return df_dqdv


def process_GITT(df, file_path, png):
    """
    Process the GITT data for a given file in parquet format
    Plot the GITT Voltage curve and the diffusion coefficient over SOC for every cycles

    Parameters:
    - file_path: Path to the file to process

    Returns:
    - df: DataFrame containing the GITT data
    """
    start_time = time.time()

    df_nested = pulse_number_GITT(df)
    results_df = global_calculation_GITT(df_nested)
    print(results_df)

    fig = plot_test_over_time(df_nested, file_path, test='GITT', pulse=True, png=png)
    # fig.show()

    fig_GITT = plot_GITT_result(results_df, file_path, column='Diffusion Coefficient', png=png)
    # fig_GITT.show()

    fig_GITT = plot_GITT_result(results_df, file_path, column='Resistance', png=png)
    # fig_GITT.show()
    # fig_GITT = plot_GITT_result(results_df, file_path, column='R_30s')
    # fig_GITT.show()
    # fig_GITT = plot_GITT_result(results_df, file_path, column='R_60s')
    # fig_GITT.show()
    # fig_GITT = plot_GITT_result(results_df, file_path, column='R_180s')
    # fig_GITT.show()

    print('GITT Time : '+str(int(time.time() - start_time))+' s')
    return df_nested


def process_ICI(df, file_path, png):
    """
    Process the ICI data for a given file in parquet format
    Plot the ICI Voltage curve and the diffusion coefficient over SOC for every cycles

    Parameters:
    - file_path: Path to the file to process

    Returns:
    - df: DataFrame containing the ICI data
    """
    start_time = time.time()

    df_nested = pulse_number_ICI(df)
    results_df = global_calculation_ICI(df_nested)
    print(results_df)

    fig = plot_test_over_time(df_nested, file_path, test='ICI', pulse=True, png=png)
    # fig.show()

    fig_GITT = plot_GITT_result(results_df, file_path, column='Diffusion Coefficient', png=png)
    # fig_GITT.show()

    fig_GITT = plot_GITT_result(results_df, file_path, column='Resistance', png=png)
    # fig_GITT.show()

    print('ICI Time : '+str(int(time.time() - start_time))+' s')
    return df_nested


def process_HPPC(df, file_path, png):
    """
    Process the HPPC data for a given file in parquet format
    Plot the HPPC Voltage curve and the diffusion coefficient over SOC for every cycles

    Parameters:
    - file_path: Path to the file to process

    Returns:
    - df: DataFrame containing the HPPC data
    """
    start_time = time.time()

    df_nested = pulse_number_HPPC(df)
    results_df = global_calculation_HPPC(df_nested)
    print(results_df)

    fig = plot_test_over_time(df_nested, file_path, test='HPPC', pulse=True, png=png)
    # fig.show()

    fig_HPPC = plot_HPPC_result(results_df, file_path, column='R', png=png)
    # fig_HPPC.show()

    fig_HPPC = plot_HPPC_result(results_df, file_path, column='P', png=png)
    # fig_HPPC.show()

    print('HPPC Time : '+str(int(time.time() - start_time))+' s')
    return df_nested

def find_test(df_input):
    start_time = time.time()
    df = df_input.copy()

    df['Test'] = 'NA'
    for cycle in df['Cycle'].unique():
        for state in ['D', 'C']:
            df_cycle = df[(df['Cycle'] == cycle) & (df['State'] == state)].copy()

            if len(df_cycle) > 5:

                discharge_pulse = (df_cycle['normcurrent'] < 0).diff().sum()
                df_cycle['discharge_pulse'] = (df_cycle['normcurrent'] < 0).diff().cumsum()
                charge_pulse = (df_cycle['normcurrent'] > 0).diff().sum()
                df_cycle['charge_pulse'] = (df_cycle['normcurrent'] > 0).diff().cumsum()

                # Remove pulses that last less than 10 data points (noise)
                for pulse in df_cycle['discharge_pulse'].unique():
                    if len(df_cycle[df_cycle['discharge_pulse'] == pulse]) < 10:
                        discharge_pulse -= 1

                for pulse in df_cycle['charge_pulse'].unique():
                    if len(df_cycle[df_cycle['charge_pulse'] == pulse]) < 10:
                        charge_pulse -= 1

                # if charge and discharge pulse, it is HPPC
                if discharge_pulse > 5 and charge_pulse > 5:
                    df.loc[(df['Cycle'] == cycle) & (df['State'] == state), 'Test'] = 'HPPC'

                # if charge or discharge pulse, it is either GITT or ICI
                elif (discharge_pulse > 5 or charge_pulse > 5):
                    df_cycle['Pulse'] = (df_cycle['normcurrent'] != 0).diff().gt(0).cumsum().ffill()
                    relaxation_time = 0
                    for pulse in df_cycle['Pulse'].unique():
                        df_pulse = df_cycle[df_cycle['Pulse'] == pulse]
                        relaxation_time += df_pulse[df_pulse['normcurrent'] == 0]['TestTime'].diff().sum()

                    # Threshold of 30% of relaxation time makes the difference between ICI and GITT
                    if relaxation_time / df_cycle['TestTime'].diff().sum() < 0.3:
                        df.loc[(df['Cycle'] == cycle) & (df['State'] == state), 'Test'] = 'ICI'
                    else:
                        df.loc[(df['Cycle'] == cycle) & (df['State'] == state), 'Test'] = 'GITT'

    # If there is a complete cycle with no pulses, the test is CCCV
    df['Group'] = (df['Test'] != df['Test'].shift()).cumsum()
    for group in df['Group'].unique():
        df_group = df[df['Group'] == group]
        if 'C' in df_group['State'].unique() and 'D' in df_group['State'].unique() and df_group['Test'].unique() == ['NA']:
            df.loc[df['Group'] == group, 'Test'] = 'CCCV'

    df['Test'] = df['Test'].replace('NA', np.nan)

    print('Find Test Time : '+str(int(time.time() - start_time))+' s')
    return df


def process_file(file_path, 
                 column_names=None, 
                 cycle=None,
                 debug_func=None,
                 png=False):
    """
    Process the data for a given file in parquet format.
    It detects the type of tests applied to the battery and process the data for each type of tests

    Parameters:
    - file_path: Path to the file to process
    - column_names (optional): Dictionary containing the column names to be used for the analysis
    The useful columns are: Voltage, Current, and SysTime (system time or test time)
    The dictionnary should look like: 
        column_names = {'time_column': 'SysTime',
                        'voltage_column': 'Voltage',
                        'current_column': 'Current',}
    - cycle (optional): List of cycle numbers to process

    Returns:
    - df: DataFrame containing the processed data
    """
    start_time = time.time()

    df = preprocessing_files(file_path, column_names, cycle, debug_func)
    df = find_test(df)

    for test in df['Test'].unique():
        if not pd.isna(test) and len(df[df['Test'] == test]) > 5:
            print('Test :', test, '; Length :', len(df[df['Test'] == test]))
            
            df_test = df[df['Test'] == test]

            if test == 'GITT':
                df_test = process_GITT(df_test, file_path, png=png)
            elif test == 'ICI':
                df_test = process_ICI(df_test, file_path, png=png)
            elif test == 'HPPC':
                df_test = process_HPPC(df_test, file_path, png=png) 
            elif test == 'CCCV' and (df['Test'].unique() == ['CCCV']).all():
                df_test = df_test[df_test['C_Rate'] > 0.2]
                df_test = process_dqdv(df_test, file_path, pocv=True, png=png)

    print('Total Time : '+str(int(time.time() - start_time))+' s')
    return df

