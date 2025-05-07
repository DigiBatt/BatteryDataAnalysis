import requests
import os
import zipfile
import shutil
import importlib.util
import sys
import json
from .processing import process_file

def scrapping_zenodo(ACCESS_TOKEN, download_folder, community):
    """Searches and downloads files from a given Zenodo community (for example "battery-knowledge-base")

    In the given folder path, each record is stored in a separate folder named with the record ID.
    Then, each data file is stored in a single folder with the same name as the file.

    Parameters
    ----------
    ACCESS_TOKEN : str
        Access token to access the Zenodo API.
    download_folder : str
        Path to the folder where the files will be downloaded.
    community : str
        Name of the Zenodo community to search for files.
    """
    file_extensions = ['.csv', '.txt', '.parquet', '.xlsx', '.xls', '.json', '.zip']
    max_size = 0.5 * 1024**3  # 0.5 GB
    useless_file_list = ['requirements.txt', 'requirements-docs.txt']  #'readme.txt'

    response = requests.get(
        "https://zenodo.org/api/records",
        params={"access_token": ACCESS_TOKEN,
                "communities": community, 
                "size": 1000,
                })

    if response.status_code == 200:
        records = response.json()["hits"]["hits"]
        print('Scrapping Zenodo', community, 'Community')

        for record in records:
            record_id = record['id']
            record_folder = os.path.join(download_folder, str(record_id))
            os.makedirs(record_folder, exist_ok=True)
            print('Paper title: ', record['metadata']['title'])
            
            if "files" in record and record['metadata']['title'] != 'Lithium Ion Battery Test Dataset for Maritime Transport INR18650-MJ1':
                for file in record["files"]:
                    file_key = file["key"]
                    file_name, file_ext = os.path.splitext(file_key)
                    # file_folder = os.path.join(record_folder, file_name)

                    file_folder = record_folder
                    file_path = os.path.join(file_folder, file_key)

                    if not os.path.exists(file_path):
                        download_url = f"https://zenodo.org/record/{record_id}/files/{file_key}?download=1"
                        response = requests.get(download_url)

                        if response.status_code == 429:
                            print('Reset time of the current rate limit:',  response.headers['X-RateLimit-Reset'])

                        elif response.status_code == 200:
                            head_response = requests.head(download_url)
                            file_size = int(head_response.headers.get("Content-Length", 0))

                            if file_ext in file_extensions and file_key not in useless_file_list:
                                # Zip Folders
                                if file_ext == ".zip":
                                    print(f"Downloading {file_key}")
                                    file_path_zip = os.path.join(record_folder, file_key)
                                    extract_folder = os.path.join(record_folder, f"{file_key}_extracted")
                                    os.makedirs(extract_folder, exist_ok=True)

                                    with open(file_path_zip, "wb") as f:
                                        for chunk in response.iter_content(chunk_size=8192):
                                            f.write(chunk)

                                    with zipfile.ZipFile(file_path_zip, "r") as zip_ref:
                                        zip_ref.extractall(extract_folder)
                                    os.remove(file_path_zip)

                                    # Keep only good extensions
                                    for root, dirs, files in os.walk(extract_folder):
                                        for file_key in files:
                                            file_name, file_ext = os.path.splitext(file_key)
                                            # file_folder = os.path.join(record_folder, file_name)

                                            file_folder = record_folder
                                            file_path = os.path.join(file_folder, file_key)

                                            if file_ext in file_extensions:
                                                if os.path.exists(file_path):
                                                    print(f"{file_key} already downloaded.")
                                                    continue
                                                
                                                else:
                                                    os.makedirs(file_folder, exist_ok=True)
                                                    source_path = os.path.join(root, file_key)
                                                    if not os.path.exists(file_path):
                                                        shutil.move(source_path, file_path)
                                    shutil.rmtree(extract_folder, ignore_errors=True)

                                # Data Files
                                elif file_size < max_size:
                                    print(f"Downloading {file_key}")
                                    os.makedirs(file_folder, exist_ok=True)
                                    with open(file_path, "wb") as f:
                                        # f.write(response.content)
                                        for chunk in response.iter_content(chunk_size=8192):
                                            f.write(chunk)
                    else:                        
                        print(f"{file_key} already downloaded.")
    else:
        print("Error :", response.status_code, response.text)


def crawl_and_process(directory_path):
    """Crawls through the given directory and processes data files

    The results are stored under HTML plots in the same folder as the data files, in a folder named as the data file name.
    To improve the processing, 2 files can be added to the folder corresponding to the data file you want to process:

    - ``column_names.json``: a JSON file with the column names to be used in the processing, under the template:

    .. code-block:: json

        {
            "time_column_name": "SysTime",
            "voltage_column_name": "Voltage",
            "current_column_name": "Current"
        }

    - ``debug_func.py``: a Python file with a function named ``debug_func`` that takes a DataFrame as input and returns a modified DataFrame, under the template:

    .. code-block:: python

        def debug_func(df):
            # Perform debugging operations on the DataFrame
            return df

    Parameters
    ----------
    directory_path : str
        Path to the directory to crawl and process
    """
    max_size = 2 * 1024**3  # 2 GB
    file_extensions = ['.csv', '.txt', '.parquet', '.xlsx', '.xls', '.json']

    for root, dirs, files in os.walk(directory_path):
        print('root', root)
        for file in files:
            file_name, file_ext = os.path.splitext(file)
            file_path = os.path.join(root, file)

            # Try loading column_names if it exists
            column_names = {}
            json_path = os.path.join(root, "column_names.json")
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    column_names = json.load(f)

            # Try loading debug_func if it exists
            debug_func = None
            debug_func_path = os.path.join(root, "debug_func.py")
            if os.path.exists(debug_func_path):
                spec = importlib.util.spec_from_file_location("debug_module", debug_func_path)
                debug_module = importlib.util.module_from_spec(spec)
                sys.modules["debug_module"] = debug_module
                spec.loader.exec_module(debug_module)

                if hasattr(debug_module, "debug_func"):
                    debug_func = getattr(debug_module, "debug_func")
            
            file_already_processed = any(d.startswith(file_name) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
            file_too_heavy = os.path.getsize(file_path) > max_size
            good_extension = file_ext in file_extensions
            if not file_already_processed and not file_too_heavy and good_extension:
                try:
                    df, df_result = process_file(file_path, column_names=column_names, debug_func=debug_func)
                except Exception as e:
                    print('Error processing file:', e)