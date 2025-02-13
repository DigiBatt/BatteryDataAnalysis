from preprocessing import preprocessing_files
from analysis_dqdv import calculate_dqdv_for_all_cycle
from analysis_GITT import pulse_number_GITT, global_calculation_GITT
from analysis_HPPC import pulse_number_HPPC, global_calculation_HPPC
from plotting import plot_dqdv_heatmap, plot_pocv, plot_DQDV_result, plot_GITT_result, plot_GITT_test, plot_HPPC_result, plot_HPPC_test

import plotly.express as px

def process_dqdv(file_path, 
                 column_names=None, 
                 cycle=None,
                 debug_func=None, 
                 curve=True, 
                 heatmap=True, 
                 pocv=False, 
                 save=False, 
                 smoothing=True):
    """
    Process the dQ/dV data for a given file in parquet format
    Plot the dQ/dV curves and the dQ/dV heatmap for every cycles
    
    Parameters:
    - file_path: Path to the file to process
    - column_names (optional): Dictionary containing the column names to be used for the analysis
    The useful columns are: Voltage, Current, Capacity, SysTime (system time or test time), Cycle and State (charging state)
    These columns should be the values of the dictionary and the keys should be the corresponding column names in the file
    - curve (optional): Boolean indicating whether to plot the dQ/dV curves
    - heatmap (optional): Boolean indicating whether to plot the dQ/dV heatmap
    - pocv (optional): Boolean indicating whether to plot the Voltage over Capacity curves
    - save (optional): Boolean indicating whether to save the plots
    - cycle (optional): List of cycle numbers to process
    - smoothing (optional): Boolean indicating whether to apply smoothing to the dQ/dV curves

    Returns:
    - df_dqdv: DataFrame containing the dQ/dV data
    """
    df = preprocessing_files(file_path, column_names, cycle, debug_func)

    df_dqdv = calculate_dqdv_for_all_cycle(df, smoothing=smoothing)

    if curve:
        fig_pocv = plot_DQDV_result(df_dqdv, file_path, save)
        fig_pocv.show()

    if heatmap:
        heatmap_dqdv = plot_dqdv_heatmap(df_dqdv, file_path, save)
        heatmap_dqdv.show()

    if pocv:
        fig_pocv = plot_pocv(df, file_path, save)
        fig_pocv.show()    

    return df_dqdv


def process_GITT(file_path, 
                 column_names=None, 
                 cycle=None,
                 debug_func=None):
    """
    Process the GITT data for a given file in parquet format
    Plot the GITT Voltage curve and the diffusion coefficient over SOC for every cycles

    Parameters:
    - file_path: Path to the file to process

    Returns:
    - df: DataFrame containing the GITT data
    """

    df = preprocessing_files(file_path, column_names, cycle, debug_func)
    df_nested = pulse_number_GITT(df)

    results_df = global_calculation_GITT(df_nested)
    print(results_df)

    fig_GITT = plot_GITT_test(df_nested, file_path)
    fig_GITT.show()

    fig_GITT = plot_GITT_result(results_df, file_path, column='Diffusion Coefficient')
    fig_GITT.show()

    fig_GITT = plot_GITT_result(results_df, file_path, column='Resistance')
    fig_GITT.show()

    return df_nested


def process_HPPC(file_path, 
                 column_names=None, 
                 cycle=None,
                 debug_func=None):
    """
    Process the HPPC data for a given file in parquet format
    Plot the HPPC Voltage curve and the diffusion coefficient over SOC for every cycles

    Parameters:
    - file_path: Path to the file to process

    Returns:
    - df: DataFrame containing the HPPC data
    """

    df = preprocessing_files(file_path, column_names, cycle, debug_func)
    df_nested = pulse_number_HPPC(df)

    results_df = global_calculation_HPPC(df_nested)
    print(results_df)

    fig_HPPC = plot_HPPC_test(df_nested, file_path)
    fig_HPPC.show()

    fig_HPPC = plot_HPPC_result(results_df, file_path, column='R')
    fig_HPPC.show()

    fig_HPPC = plot_HPPC_result(results_df, file_path, column='P')
    fig_HPPC.show()

    return df_nested


