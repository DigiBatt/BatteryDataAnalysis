import os

from preprocessing import preprocessing_files
from analysis import calculate_dqdv_for_all_cycle
from plotting import plot_dqdv, plot_dqdv_heatmap, plot_pocv

import plotly.express as px
import pyarrow.parquet as pq

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