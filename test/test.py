import os
import pyarrow.parquet as pq
import plotly.express as px

from preprocessing import preprocessing_files, standardize_column_names, charging_state, process_useful_columns
from analysis import calculate_dqdv_for_all_cycle
from plotting import plot_dqdv, plot_dqdv_heatmap, plot_pocv
from processing import process_dqdv

script_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(script_dir, 'files')

for file_name in os.listdir(folder_path):
    if file_name.endswith('.parquet'):
        file_path = os.path.join(folder_path, file_name)

        table = pq.read_table(file_path)
        df = table.to_pandas()


        # Test Preprocessing
        print(df.columns)
        df_standard = standardize_column_names(df)
        print(df_standard.columns)

        df_state = charging_state(df_standard)

        print(df_state.head())

        df_useful = process_useful_columns(df_state)
        print(df_useful.head())

        df_preprocessed = preprocessing_files(file_path, column_names=None)
        print(df_preprocessed.head())


        # Test Analysis
        df_dqdv = calculate_dqdv_for_all_cycle(df_preprocessed)
        print(df_dqdv.head())

        fig = px.scatter(df_dqdv, x='smoothed_voltage', y='smoothed_dqdv', color='Cycle', title=file_name)
        fig.show()


        # Test Plotting
        fig_dqdv = plot_dqdv(df_dqdv, file_path)
        fig_dqdv.show()

        fig_heatmap_dqdv = plot_dqdv_heatmap(df_dqdv, file_path)
        fig_heatmap_dqdv.show()

        fig_pocv = plot_pocv(df_dqdv, file_path)
        fig_pocv.show()


        # Test Processing
        df = process_dqdv(file_path)  

        