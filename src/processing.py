from preprocessing import preprocessing_files
from analysis_dqdv import calculate_dqdv_for_all_cycle
from analysis_GITT import pulse_number_GITT, global_calculation_GITT
from analysis_HPPC import pulse_number_HPPC, global_calculation_HPPC
from analysis_ICI import pulse_number_ICI, global_calculation_ICI
from plotting import *

import time
import pandas as pd

def process_file(file_path, 
                 column_names=None, 
                 cycle=None,
                 debug_func=None,
                 save=False,
                 png=False):
    """Processes the data for a given file in .parquet or .csv format

    It detects the type of tests applied to the battery and process the data for each type of tests

    Parameters
    ----------
    file_path : str
        Path to the file to process
    column_names : Dict, optional
        Dictionary containing the column names to be used for the analysis
        The useful columns are: Voltage, Current, and SysTime (system time or test time) that should be values and the real column names should be their keys
    cycle : int, optional
        List of cycle numbers to process
    debug_func : function, optional
        Function to be called during the preprocessing of the data to debug a file
    save : bool, optional
        Whether to save the result plots in format .html (default is False).
    png : bool, optional
        Whether to save the result plots as png files (default is False).

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the processed data

    Examples
    --------
    >>> column_names={'time_column': 'SysTime', 'voltage_column': 'Voltage', 'current_column': 'Current'}
    >>> df = process_file(file_path, column_names=column_names)
    """
    start_time = time.time()

    df = preprocessing_files(file_path, column_names, cycle, debug_func)
    df = find_test(df)

    for test in df['Test'].unique():
        if not pd.isna(test) and len(df[df['Test'] == test]) > 5:
            print('Test :', test, '; Length :', len(df[df['Test'] == test]))
            
            df_test = df[df['Test'] == test]

            if test == 'GITT':
                df_test = process_GITT(df_test, file_path, save=save, png=png)

            elif test == 'ICI':
                df_test = process_ICI(df_test, file_path, save=save, png=png)

            elif test == 'HPPC':
                df_test = process_HPPC(df_test, file_path, save=save, png=png) 

            elif test == 'CCCV' and (df['Test'].unique() == ['CCCV']).all():
                df_test = df_test[df_test['C_Rate'] > 0.2]
                df_test = process_dqdv(df_test, file_path, save=save, png=png)

    print('Total Time : '+str(int(time.time() - start_time))+' s')
    return df


def find_test(df_input):
    """Finds the test applied to the battery

    Analyses the test type for each cycle according to its current pulses number
    It needs the preprocessed data with in particular the columns Cycle, State and normcurrent

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the preprocessed data

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the HPPC data
    """
    start_time = time.time()
    df = df_input.copy()

    df['Test'] = 'NA'
    for cycle in df['Cycle'].unique():
        for state in ['D', 'C']:
            df_cycle = df[(df['Cycle'] == cycle) & (df['State'] == state)].copy()

            if len(df_cycle) > 5:

                # for pulse in ['discharge_pulse', 'charge_pulse']:
                #     pulse_number_list = []
                #     pulse_sign = 1 if pulse == 'discharge_pulse' else -1
                #     pulse_count = (pulse_sign * df_cycle['normcurrent'] < 0).diff().sum()
                #     df_cycle[pulse] = (pulse_sign * df_cycle['normcurrent'] < 0).diff().cumsum()

                #     for pulse_number in df_cycle[pulse].unique():
                #         if len(df_cycle[df_cycle[pulse] == pulse_number]) < 10:
                #             pulse_count -= 1
                #     pulse_number_list.append(pulse_count)

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


def process_dqdv(df, 
                 file_path,
                 heatmap=False,
                 save=False,
                 png=False):
    """Process the CCCV data for a given preprocessed file

    Plots the dQ/dV curves and the dQ/dV heatmap for every cycles
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the CCCV data to process.
    file_path : str
        Path to the file to process.
    heatmap : bool, optional
        Whether to plot the dQ/dV heatmap (default is False).
    save : bool, optional
        Whether to save the plots in format .html(default is False).
    png : bool, optional
        Whether to save the plots as png files (default is False).

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the dQ/dV data.
    """
    start_time = time.time()

    df_dqdv = calculate_dqdv_for_all_cycle(df)

    if df_dqdv is not False:
        fig = plot_test_over_time(df, file_path, save=save, png=png)
        fig.show()

        fig_dqdv = plot_DQDV_result(df_dqdv, file_path, save=save, png=png)
        fig_dqdv.show()

        fig_pocv = plot_pocv(df, file_path, save=save, png=png)
        fig_pocv.show()

        if heatmap:
            heatmap_dqdv = plot_dqdv_heatmap(df_dqdv, file_path, save=save, png=png)
            heatmap_dqdv.show()
    
    print('dQ/dV Time : '+str(int(time.time() - start_time))+' s')
    return df_dqdv


def process_GITT(df, 
                 file_path, 
                 save=False,
                 png=False):
    """Processes the GITT data for a given preprocessed file

    Plots the GITT Voltage curve and the diffusion coefficient over SOC for every cycles

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the GITT data to process
    file_path : str
        Path to the file to process
    save : bool, optional
        Whether to save the plots in format .html(default is False).
    png : bool, optional
        Whether to save the plots as png files (default is False).

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the GITT data
    """
    start_time = time.time()

    df_nested = pulse_number_GITT(df)
    results_df = global_calculation_GITT(df_nested)
    print(results_df)

    fig = plot_test_over_time(df_nested, file_path, test='GITT', pulse=True, save=save, png=png)
    fig.show()

    fig_GITT = plot_GITT_result(results_df, file_path, column='Diffusion Coefficient', save=save, png=png)
    fig_GITT.show()

    fig_GITT = plot_GITT_result(results_df, file_path, column='Resistance', save=save, png=png)
    # fig_GITT.show()
    # fig_GITT = plot_GITT_result(results_df, file_path, column='R_30s', save=save)
    # fig_GITT.show()
    # fig_GITT = plot_GITT_result(results_df, file_path, column='R_60s', save=save)
    # fig_GITT.show()
    # fig_GITT = plot_GITT_result(results_df, file_path, column='R_180s', save=save)
    # fig_GITT.show()

    print('GITT Time : '+str(int(time.time() - start_time))+' s')
    return df_nested


def process_ICI(df, 
                file_path, 
                save=False,
                png=False):
    """Processes the ICI data for a given preprocessed file

    Plots the ICI Voltage curve and the diffusion coefficient over SOC for every cycles

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the ICI data to process
    file_path : str
        Path to the file to process
    save : bool, optional
        Whether to save the plots in format .html(default is False).
    png : bool, optional
        Whether to save the plots as png files (default is False).

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the ICI data
    """
    start_time = time.time()

    df_nested = pulse_number_ICI(df)
    results_df = global_calculation_ICI(df_nested)
    print(results_df)

    fig = plot_test_over_time(df_nested, file_path, test='ICI', pulse=True, save=save, png=png)
    fig_GITT = plot_GITT_result(results_df, file_path, column='Diffusion Coefficient', save=save, png=png)
    fig_GITT = plot_GITT_result(results_df, file_path, column='Resistance', save=save, png=png)

    print('ICI Time : '+str(int(time.time() - start_time))+' s')
    return df_nested


def process_HPPC(df, 
                 file_path, 
                 save=False,
                 png=False):
    """Processes the HPPC data for a given preprocessed file

    Plots the HPPC Voltage curve and the diffusion coefficient over SOC for every cycles

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the HPPC data to process
    file_path : str
        Path to the file to process
    save : bool, optional
        Whether to save the plots in format .html(default is False).
    png : bool, optional
        Whether to save the plots as png files (default is False).

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the HPPC data
    """
    start_time = time.time()

    df_nested = pulse_number_HPPC(df)
    results_df = global_calculation_HPPC(df_nested)
    print(results_df)

    fig = plot_test_over_time(df_nested, file_path, test='HPPC', pulse=True, save=save, png=png)
    fig_HPPC = plot_HPPC_result(results_df, file_path, column='R', save=save, png=png)
    fig_HPPC = plot_HPPC_result(results_df, file_path, column='P', save=save, png=png)

    print('HPPC Time : '+str(int(time.time() - start_time))+' s')
    return df_nested
