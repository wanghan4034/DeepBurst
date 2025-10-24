import os
from tqdm import tqdm
import pandas as pd 
import subprocess
from src.data.features.rawdata import RawDataFileInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.data.features.constants import EXPERIMENT_TARGETS, TOTAL_ASSAYS, TARGET_CELL_TYPES

def download(urls, dir):

    def _func(url):
        file_name = url.split('/')[-1]
        save_path = os.path.join(dir, file_name)
        print("save_path:", save_path)
        process = subprocess.Popen(['wget', '-O', save_path,'-c', url],stdout=subprocess.PIPE)
        for _, line in enumerate(process.stdout):
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith('Length:'):
                total_size = int(decoded_line.split()[1])
            elif decoded_line.startswith('%'):
                progress = int(decoded_line.split()[0].replace('%', ''))
                tqdm.update(progress)

        process.stdout.close()
        process.wait()
    

    with ThreadPoolExecutor(max_workers=5) as executor: 
        future_to_url = {executor.submit(_func, url): url for url in urls}

        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                data = future.result()
            except Exception as exc:
                print(f'{url} error: {exc}')
            else:
                print(f'{url} success')

def get_all_urls(url_file, rawdata_dir): 
    raw_data_file_info = RawDataFileInfo(url_file, rawdata_dir)
    cell_types= TARGET_CELL_TYPES
    assays = TOTAL_ASSAYS
    exprimental_targets = EXPERIMENT_TARGETS
    data = raw_data_file_info.get_file_index(cell_types=cell_types,assays = assays, exprimental_targets=exprimental_targets,reduce_replicate=False, is_download=True)
    urls = [file_info.url for file_info in data]
    return urls


if __name__ == '__main__':
    url_file = 'extra/datasets/epigenetic/hg38/media-3.csv'
    # rawdata_dir = 'extra/datasets/rawdata'
    rawdata_dir = 'extra/datasets/epigenetic/hg38'
    urls = get_all_urls(url_file,rawdata_dir)
    download(urls=urls,dir=rawdata_dir)