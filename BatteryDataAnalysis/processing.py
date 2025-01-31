from preprocessing import preprocessing_files
from analysis_dqdv import calculate_dqdv_for_all_cycle
from analysis_GITT import pulse_number, global_calculation
from plotting import plot_dqdv, plot_dqdv_heatmap, plot_pocv, plot_GITT_result, plot_GITT_test

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

    df = preprocessing_files(file_path)
    df = pulse_number(df)

    results_df = global_calculation(df)

    fig_GITT = plot_GITT_test(df, file_path)
    fig_GITT.show()

    fig_GITT = plot_GITT_result(results_df, file_path, method=1)
    fig_GITT.show()

    fig_GITT = plot_GITT_result(results_df, file_path, method=2)
    fig_GITT.show()

    fig_GITT = plot_GITT_result(results_df, file_path, method=3)
    fig_GITT.show()

    return df


file = 'GITT_AG4_S_1577' # GITT_AG4_S_1577, GITT_S_333, GITT_S_170, GITT_LACB440BP2
folder_path = "C:/Users/edgarl/OneDrive - SINTEF/Documents/Test/cold_test/parquet_files_testing/files"
file_path = os.path.join(folder_path, f"{file}.parquet")
process_GITT(file_path)



# file = 'estimation_results'
# folder_path = "C:/Users/edgarl/OneDrive - SINTEF/Documents/Test/cold_test/parquet_files_testing/files/GITT estimation results"
# # folder_path = "C:/Users/edgarl/OneDrive - SINTEF/Documents/Test/cold_test/parquet_files_testing/files/GITT data"
# file_path_json = os.path.join(folder_path, f"{file}.json")
# file_path_csv = os.path.join(folder_path, f"{file}.csv")

# df = pd.read_json(file_path_json)

# df = pd.DataFrame(df['inferred parameters'].tolist()) #.tolist()
# print(df)

# fig_GITT = px.scatter(
#     x=df.index,
#     y=df['Positive electrode diffusivity [m2.s-1]'],
#     title='GITT Experimental Positive Electrode diffusivity coefficient'
# )
# fig_GITT.show()

# Convertir en CSV
# df.to_csv(file_path_csv, index=False, encoding="utf-8")