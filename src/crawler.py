import requests
import os
import zipfile
import shutil
import time
from processing import process_file

def scrapping_zenodo():
    ACCESS_TOKEN = '70zVqGpbBA0ueu68VJ2s02zNzRJKW7Kpju1xmj4n4oUoO5GT79XDx1FDyc8I'
    community = "battery-knowledge-base"
    download_folder = 'C:/Users/edgarl/OneDrive - SINTEF/Documents/Zenodo_crawling'
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
            
            if "files" in record:
                for file in record["files"]:
                    file_key = file["key"]
                    file_name, file_ext = os.path.splitext(file_key)
                    file_folder = os.path.join(record_folder, file_name)
                    file_path = os.path.join(file_folder, file_key)

                    if not os.path.exists(file_path):
                        download_url = f"https://zenodo.org/record/{record_id}/files/{file_key}?download=1"
                        response = requests.get(download_url)
                        print('Requests remaining:', response.headers['X-RateLimit-Remaining'])
                        time.sleep(1)

                        if response.status_code == 429:
                            print('Reset time of the current rate limit:',  response.headers['X-RateLimit-Reset'])

                        elif response.status_code == 200:
                            print('response_statut == 200')
                            head_response = requests.head(download_url)
                            file_size = int(head_response.headers.get("Content-Length", 0))

                            if file_ext in file_extensions and file_key not in useless_file_list:
                                print('file_ext in file_extensions')

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
                                            file_folder = os.path.join(record_folder, file_name)
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


def crawl_and_process():
    directory_path = 'C:/Users/edgarl/OneDrive - SINTEF/Documents/Zenodo_crawling'
    file_extensions = ['.csv', '.txt', '.parquet', '.xlsx', '.xls']
    max_size = 0.5 * 1024**3  # 0.5 GB

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # print('root', root)
            # print('files', files)
            file_name, file_ext = os.path.splitext(file)
            # file_folder = os.path.join(root, file_name)
            file_path = os.path.join(root, file)
            if not any(f.endswith('.html') for f in os.listdir(root)):
                if file_ext in file_extensions and os.path.getsize(file_path) < max_size:
                    try:
                        df, df_result = process_file(file_path, save=True)
                    except Exception as e:
                        print('Error processing file:', e)

# scrapping_zenodo()
# crawl_and_process()
