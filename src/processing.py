from preprocessing import read_file
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
                 my_func_list=[],
                 input='path',
                 save=True,
                 png=False,
                 plot=True):
    """Processes the data for a given file

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
    my_func_list : list, optional
        List of functions to add to the calculation process
    save : bool, optional
        Whether to save the result plots in format .html (default is False).
    png : bool, optional
        Whether to save the result plots as png files (default is False).

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing the processed data.
    result_dict : dict
        Dictionary containing the processed results for each test type.
        Keys are test names ('GITT', 'ICI', 'HPPC') and values are DataFrames with the respective results for each test.

    Examples
    --------
    Basic usage:

    >>> df = process_file(file_path)
    File : GITT_AG4_S_1577.parquet
    Length : 9154638
    Preprocessing Time : 41s
    Find Test Time : 48s
    Test : GITT ; Length : 9061575
        Pulse  Cycle   TestTime       SOC       OCV    Ipulse  Diffusion Coefficient  Resistance      R_1s      R_30s      R_60s     R_180s      Tau    State
    Pulse      0      1   46801.00  1.000000  2.484417 -0.043325           3.224942e-15   24.730671  6.169087  28.370994  30.130583  36.914223  3598.98     D
    Pulse      1      1   53999.98  0.980011  0.575001 -0.043325           4.672011e-16    1.997885  1.474679   2.089985   2.411197   3.278645  3599.98     D
    Pulse      2      1   61199.98  0.960011  0.379431 -0.043325           3.873728e-16    1.412661  1.106010   1.468100   1.685784   2.282023  3599.98     D
    Pulse      3      1   68399.98  0.940010  0.272730 -0.043324           2.772498e-16    1.205467  0.979083   1.249654   1.411445   1.843224  3599.98     D
    Pulse      4      1   75599.98  0.920010  0.216829 -0.043325           6.646707e-17    1.077636  0.915281   1.112744   1.224103   1.524546  3599.98     D
    ...      ...    ...        ...       ...       ...       ...                    ...         ...       ...        ...        ...        ...      ...   ...
    Pulse    133      2  962099.98  0.136391  0.087613 -0.043325           1.887492e-19    0.220889  0.143877   0.239615   0.273980   0.321392  3599.98     D
    Pulse    134      2  969299.98  0.111716  0.087493 -0.043325           2.520212e-19    0.225816  0.147635   0.244767   0.279301   0.326996  3599.98     D
    Pulse    135      2  976499.98  0.087041  0.087352 -0.043325           3.536782e-19    0.231572  0.152126   0.251083   0.285541   0.333647  3599.98     D
    Pulse    136      2  983699.98  0.062366  0.087180 -0.043325           5.092976e-19    0.238380  0.157598   0.258096   0.292359   0.341250  3599.98     D
    Pulse    137      2  990899.98  0.037691  0.086971 -0.043325           8.648767e-19    0.246495  0.163909   0.266569   0.300817   0.350488  3599.98     D
    [138 rows x 15 columns]
    GITT Time : 21s
    Total Time : 115s

    In addition, you can add a column_names dictionnary and a debug function if needed:

    >>> column_names={'original_time_column': 'SysTime', 
    ...               'original_voltage_column': 'Voltage', 
    ...               'original_current_column': 'Current'}
    >>> def my_debug_func(df):
    ...     df = df[df['original_time_column'] < 1e6]
    ...     return df
    >>> df = process_file(file_path, column_names=column_names, debug_func=my_debug_func)
    File : GITT_AG4_S_1577.parquet
    Length : 9154638
    Preprocessing Time : 41s
    Find Test Time : 48s
    Test : GITT ; Length : 9061575
        Pulse  Cycle   TestTime       SOC       OCV    Ipulse  Diffusion Coefficient  Resistance      R_1s      R_30s      R_60s     R_180s      Tau    State
    Pulse      0      1   46801.00  1.000000  2.484417 -0.043325           3.224942e-15   24.730671  6.169087  28.370994  30.130583  36.914223  3598.98     D
    Pulse      1      1   53999.98  0.980011  0.575001 -0.043325           4.672011e-16    1.997885  1.474679   2.089985   2.411197   3.278645  3599.98     D
    Pulse      2      1   61199.98  0.960011  0.379431 -0.043325           3.873728e-16    1.412661  1.106010   1.468100   1.685784   2.282023  3599.98     D
    Pulse      3      1   68399.98  0.940010  0.272730 -0.043324           2.772498e-16    1.205467  0.979083   1.249654   1.411445   1.843224  3599.98     D
    Pulse      4      1   75599.98  0.920010  0.216829 -0.043325           6.646707e-17    1.077636  0.915281   1.112744   1.224103   1.524546  3599.98     D
    ...      ...    ...        ...       ...       ...       ...                    ...         ...       ...        ...        ...        ...      ...   ...
    Pulse    133      2  962099.98  0.136391  0.087613 -0.043325           1.887492e-19    0.220889  0.143877   0.239615   0.273980   0.321392  3599.98     D
    Pulse    134      2  969299.98  0.111716  0.087493 -0.043325           2.520212e-19    0.225816  0.147635   0.244767   0.279301   0.326996  3599.98     D
    Pulse    135      2  976499.98  0.087041  0.087352 -0.043325           3.536782e-19    0.231572  0.152126   0.251083   0.285541   0.333647  3599.98     D
    Pulse    136      2  983699.98  0.062366  0.087180 -0.043325           5.092976e-19    0.238380  0.157598   0.258096   0.292359   0.341250  3599.98     D
    Pulse    137      2  990899.98  0.037691  0.086971 -0.043325           8.648767e-19    0.246495  0.163909   0.266569   0.300817   0.350488  3599.98     D
    [138 rows x 15 columns]
    GITT Time : 21s
    Total Time : 115s
    """
    start_time = time.time()

    df_list, file_path_list = read_file(file_path, column_names, cycle, debug_func, input)
    result_dict_list = []
    for i, df in enumerate(df_list):
        new_file_path = file_path_list[i]

        base_name = os.path.splitext(os.path.basename(new_file_path))[0]
        if "- sheet " in base_name and new_file_path != file_path:
            sheet_name = base_name.split("- sheet ")[-1]
            print('\nSheet', sheet_name)

        df = find_test(df)
        fig_global = plot_test_over_time(df, new_file_path, save, png, plot, test='global_file')

        result_dict = {}
        for test in df['Test'].unique():
            print(test)
            if len(df[df['Test'] == test]) > 5: #not pd.isna(test) and 
                print('Test :', test, '; Length :', len(df[df['Test'] == test]))
                
                df_test = df[df['Test'] == test]

                if test == 'GITT':
                    df_test, GITT_result_df = process_GITT(df_test, new_file_path, my_func_list, save, png, plot)
                    result_dict['GITT'] = GITT_result_df

                elif test == 'ICI':
                    df_test, ICI_result_df = process_ICI(df_test, new_file_path, my_func_list, save, png, plot)
                    result_dict['ICI'] = ICI_result_df

                elif test == 'HPPC':
                    df_test, HPPC_result_df = process_HPPC(df_test, new_file_path, my_func_list, save, png, plot)
                    result_dict['HPPC'] = HPPC_result_df

                elif test == 'CCCV' and (df['Test'].unique() == ['CCCV']).all():
                    df_test = df_test[df_test['C_Rate'] > 0.2]
                    df_test = process_dqdv(df_test, new_file_path, save, png, plot)

                elif test == 'NA' and (df['Test'].unique() == ['NA']).all():
                    # df_test = df_test[df_test['C_Rate'] > 0.2]
                    df_test = process_dqdv(df_test, new_file_path, save, png, plot)

                # else:
                #     df_test = process_dqdv(df_test, new_file_path, save, png, plot)

                result_dict_list.append(result_dict)

    print('Total Time : '+str(int(time.time() - start_time))+'s')
    if len(df_list) == 1:
        return df_list[0], result_dict_list[0]
    else:
        return df_list, result_dict_list


def find_test(df_input):
    """Finds the test applied to the battery

    Analyses the test type for each cycle according to its current pulses number
    It needs the preprocessed data with in particular the columns Cycle, State and normcurrent

    Parameters
    ----------
    df_input : pandas.DataFrame
        DataFrame containing the preprocessed data

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the preprocessed data.
    """
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

                else:
                    volt_diff = df_cycle['Voltage'].diff()

    # If there is a complete cycle with no pulses, the test is CCCV
    # df['Group'] = (df['Test'] != df['Test'].shift()).cumsum()
    # for group in df['Group'].unique():
    #     df_group = df[df['Group'] == group]
    #     if 'C' in df_group['State'].unique() and 'D' in df_group['State'].unique() and df_group['Test'].unique() == ['NA']:
    #         df.loc[df['Group'] == group, 'Test'] = 'CCCV'

    # df['Test'] = df['Test'].replace('NA', np.nan)

    print('Find Test Time : '+str(int(time.time() - start_time))+'s')
    return df


def process_dqdv(df, 
                 file_path,
                 save,
                 png,
                 plot,
                 heatmap=False):
    """Processes the CCCV data for a given preprocessed file

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
        fig = plot_test_over_time(df, file_path, save, png, plot)
        fig_dqdv = plot_DQDV_result(df_dqdv, file_path, save, png, plot)
        fig_pocv = plot_pocv(df, file_path, save, png, plot)

        if heatmap:
            heatmap_dqdv = plot_dqdv_heatmap(df_dqdv, file_path, save, png, plot)
    
    print('dQ/dV Time : '+str(int(time.time() - start_time))+'s')
    return df_dqdv


def process_GITT(df, 
                 file_path,
                 my_func_list,
                 save,
                 png,
                 plot):
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
    df_nested : dict
        Dict containing pulses as keys and their respective DataFrames as values
    results_df : pandas.DataFrame
        DataFrame containing the GITT parameters structured by pulse number
    """
    start_time = time.time()

    df_nested = pulse_number_GITT(df)
    results_df = global_calculation_GITT(df_nested, my_func_list)
    print(results_df)

    try:
        fig = plot_test_over_time(df_nested, file_path, save, png, plot, test='GITT', pulse=True)
        fig_GITT = plot_GITT_result(results_df, file_path, save, png, plot, column='Diffusion Coefficient')
        fig_GITT = plot_GITT_result(results_df, file_path, save, png, plot, column='Resistance')


        fig_GITT = plot_GITT_result(results_df, file_path, save, png, plot, column='OCV')
    except:
        print('Not possible to calculate GITT data')



    # fig_GITT = plot_GITT_result(results_df, file_path, column='R_30s', save=save)
    # fig_GITT = plot_GITT_result(results_df, file_path, column='R_60s', save=save)
    # fig_GITT = plot_GITT_result(results_df, file_path, column='R_180s', save=save)

    print('GITT Time : '+str(int(time.time() - start_time))+'s')
    return df_nested, results_df


def process_ICI(df, 
                file_path,
                my_func_list,
                save,
                png,
                plot):
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
    df_nested : dict
        Dict containing pulses as keys and their respective DataFrames as values
    results_df : pandas.DataFrame
        DataFrame containing the ICI parameters structured by pulse number
    """
    start_time = time.time()

    df_nested = pulse_number_ICI(df)
    results_df = global_calculation_ICI(df_nested, my_func_list)
    print(results_df)

    try:
        fig = plot_test_over_time(df_nested, file_path, save, png, plot, test='ICI', pulse=True)
        fig_ICI = plot_GITT_result(results_df, file_path, save, png, plot, column='Diffusion Coefficient', test='ICI')
        fig_ICI = plot_GITT_result(results_df, file_path, save, png, plot, column='Resistance', test='ICI')
    except:
        print('Not possible to calculate ICI data')

    print('ICI Time : '+str(int(time.time() - start_time))+'s')
    return df_nested, results_df


def process_HPPC(df, 
                 file_path,
                 my_func_list,
                 save,
                 png,
                 plot):
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
    df_nested : dict
        Dict containing pulses as keys and their respective DataFrames as values
    results_df : pandas.DataFrame
        DataFrame containing the HPPC parameters structured by pulse number
    """
    start_time = time.time()

    df_nested = pulse_number_HPPC(df)
    results_df = global_calculation_HPPC(df_nested, my_func_list)
    print(results_df)

    try:
        fig = plot_test_over_time(df_nested, file_path, save, png, plot, test='HPPC', pulse=True)
        fig_HPPC = plot_HPPC_result(results_df, file_path, save, png, plot, column='R')
        fig_HPPC = plot_HPPC_result(results_df, file_path, save, png, plot, column='P')
    except:
        print('Not possible to calculate HPPC data')

    print('HPPC Time : '+str(int(time.time() - start_time))+'s')
    return df_nested, results_df
