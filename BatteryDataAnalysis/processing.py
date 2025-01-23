from preprocessing import preprocessing_files
from BatteryDataAnalysis.analysis_dqdv import calculate_dqdv_for_all_cycle
from analysis_GITT import pulse_number, global_calculation
from plotting import plot_dqdv, plot_dqdv_heatmap, plot_pocv, plot_GITT

import pyarrow.parquet as pq
import os

def process_dqdv(file_path, 
                 column_names=None, 
                 curve=True, 
                 heatmap=True, 
                 pocv=False, 
                 save=False, 
                 cycle=None,
                 smoothing=True):
    """
    Process the dQ/dV data for a given file in parquet format
    Plot the dQ/dV curves and the dQ/dV heatmap for every cycles
    
    Parameters:
    - file_path: Path to the file to process
    - column_names (optional): Dictionary containing the column names to be used for the analysis
    The useful columns are: Voltage, Current, Capacity, SysTime (system time or test time), Cycle and State (charging state)
    These columns should be the keys of the dictionary and the values should be the corresponding column names in the file
    - curve (optional): Boolean indicating whether to plot the dQ/dV curves
    - heatmap (optional): Boolean indicating whether to plot the dQ/dV heatmap
    - pocv (optional): Boolean indicating whether to plot the Voltage over Capacity curves
    - save (optional): Boolean indicating whether to save the plots
    - cycle (optional): List of cycle numbers to process
    - smoothing (optional): Boolean indicating whether to apply smoothing to the dQ/dV curves

    Returns:
    - df_dqdv: DataFrame containing the dQ/dV data
    """
    df = preprocessing_files(file_path, column_names, cycle)

    df_dqdv = calculate_dqdv_for_all_cycle(df, smoothing=smoothing)

    if curve:
        fig_dqdv = plot_dqdv(df_dqdv, file_path, save)
        fig_dqdv.show()

    if heatmap:
        heatmap_dqdv = plot_dqdv_heatmap(df_dqdv, file_path, save)
        heatmap_dqdv.show()

    if pocv:
        fig_pocv = plot_pocv(df, file_path, save)
        fig_pocv.show()

    return df_dqdv


def process_GITT(file_path):
    table = pq.read_table(file_path)
    df = table.to_pandas()

    df = pulse_number(df)

    df = df[(df['Pulse'] >= 1) & (df['Pulse'] <= 100)]

    df, results_df = global_calculation(df)

    df = df.iloc[::100]

    fig_GITT = plot_GITT(df, file_path)
    fig_GITT.show()

    return df


file = 'GITT_AG4_S_1577'
folder_path = "C:/Users/edgarl/OneDrive - SINTEF/Documents/Test/cold_test/parquet_files_testing/files"
file_path = os.path.join(folder_path, f"{file}.parquet")
process_GITT(file_path)