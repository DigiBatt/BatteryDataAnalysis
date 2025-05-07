from processing import process_file
# from crawler import crawl_and_process

column_names={}
def debug_func(df):
    df[['Time', 'ClimaTemp']] = df['Time'].str.extract(r'(.+ \d{2}:\d{2}:\d{2})\.?(\d+)?')
    return df

# file_path = 'C:/Users/edgarl/OneDrive - SINTEF/Documents/Test/cold_test/parquet_files_testing/Andres/Intelligent1/20240806_SINTEF_Halfcell_SiGr/gitt/AG4-S-1577_9_ShortLandt_06_4_20240620153129.parquet'
# file_path = "C:/Users/edgarl/OneDrive - SINTEF/Documents/Test/cold_test/parquet_files_testing/files/GITT/GITT_AG4_S_1577.parquet"
# file_path = "C:/Users/edgarl/OneDrive - SINTEF/Documents/Test/cold_test/parquet_files_testing/files/dqdv/INT_WP1_S_170_1_023_3.parquet"
file_path = "C:/Users/edgarl/OneDrive - SINTEF/Documents/Test/cold_test/parquet_files_testing/files/HPPC/Batch1/HPPC_Samsung_01.csv"
# file_path = "C:/Users/edgarl/OneDrive - SINTEF/Documents/Test/cold_test/parquet_files_testing/files/ICI/zenodo/ICI_validation/HL_CCI811_43V_ICI_EIS_3E182_03_MB_C16.parquet"

# file_path = 'C:/Users/edgarl/OneDrive - SINTEF/Documents/Test/cold_test/parquet_files_testing/files/GITT/GITT_Frode/SUM_SP5_AG4_PL_nr5.parquet'
# file_path = "C:/Users/edgarl/OneDrive - SINTEF/Documents/Test/cold_test/parquet_files_testing/files/GITT/GITT_Killyan/INT1_LNMO_GITT_4_ShortLandt_07_8.parquet"


# file_path = "C:/Users/edgarl/Downloads/sintef__melasta-slpba842124hv-2024-10-23-15077312__rate-testing.bdf.parquet"pip ion
# column_names={'time/s': 'SysTime',
#               'Ewe/V': 'Voltage',
#               'I/mA': 'Current'}

column_names = {
    "Time": "SysTime",
    "E": "Current",
    " W": "Voltage"
}

df, result_dict = process_file(file_path, column_names=column_names, debug_func=debug_func, save=True, plot=True)


# directory_path = 'C:/Users/edgarl/OneDrive - SINTEF/Test Eibar'
# crawl_and_process(directory_path)
