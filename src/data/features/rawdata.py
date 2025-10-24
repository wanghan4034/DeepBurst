import os
import pandas as pd
from functools import reduce
from copy import deepcopy
from typing import List
from src.data.features.constants import EXPRESSION_ASSAY, ASSAY_EXPERIMENT_TARGETS_MAP, URL_FILE, RAW_DATA_DIR, DNASE_SEQ


class FileInfo:
    def __init__(self, file_name:str, cell_type:str, assay:str, exprimental_target:str = None, file_path:str = None, url:str = None, s3_url:str = None,*args, **kwargs):
        self.file_name = file_name
        self.cell_type = cell_type
        self.file_path = file_path
        self.assay = assay
        self.experiment_target = exprimental_target
        self.url = url 
        self.s3_url = s3_url
    
    def __repr__(self) -> str:
        return f"{self.file_name} {self.cell_type} {self.file_path} {self.assay} {self.experiment_target}"


def item_check(origin_sets:List, target_sets:List):
    """
    check if all items in target_sets belong to origin_sets 
    """
    for item in target_sets:
        if item not in origin_sets:
            raise ValueError(f'No item {item}')

class RawDataFileInfo:
    def __init__(self,origin_url_file, raw_data_dir):
        self.data = self._read_info(origin_url_file,raw_data_dir) 
    
    def _read_info(self,origin_url_file,raw_data_dir):
        data = pd.read_csv(origin_url_file)[['Assay','Biosample term name','File download URL','Biological replicate(s)','Experiment target','s3_uri']]
        data.columns = ['assay', 'cell_type','url','replicate','experiment_target','s3_url']
        data['file_name'] = data.apply(lambda row: row['url'].split('/')[-1], axis=1)
        data['file_path'] = data.apply(lambda row: os.path.join(raw_data_dir, row['file_name']),axis=1)
        return data
    
    def get_file_index(self,cell_types:List[str] = None, assays:List[str] = None, exprimental_targets:List[str] = None, reduce_replicate:bool = True, is_download: bool = False)->List[FileInfo]:
        """
        @task : epigenetic_modification, expression, None 
        """
        if isinstance(assays, List):
            expression_assays = [assay for assay in assays if assay in EXPRESSION_ASSAY]

        data = deepcopy(self.data)
        if cell_types:
            data = data[data['cell_type'].isin(cell_types)]

        if assays:
            if not is_download:
                item_check(list(ASSAY_EXPERIMENT_TARGETS_MAP.keys()), assays)
            data = data[data['assay'].isin(assays)]

        if exprimental_targets:
            origin_expriment_targets = list(reduce(lambda x,y:x+y,ASSAY_EXPERIMENT_TARGETS_MAP.values()))
            if not is_download:
                item_check(origin_expriment_targets, exprimental_targets)
            
            if DNASE_SEQ in assays:
                data = data[data['experiment_target'].isin(exprimental_targets) | data['assay'].isin(expression_assays+[DNASE_SEQ])]
            else:           
                data = data[data['experiment_target'].isin(exprimental_targets) | data['assay'].isin(expression_assays)]

        # reduce the biosample replicate
        if reduce_replicate:
            data = data[data['replicate']=='1']

        file_infos = data.apply(lambda row:FileInfo(
                row['file_name'],
                row['cell_type'],
                row['assay'],
                row['experiment_target'],
                row['file_path'],
                row['url'],
                row['s3_url']
            ),
            axis=1
        )

        file_infos = list(file_infos) if not file_infos.empty else []

        return file_infos
    
    def get_all_cell_types(self):
        return list(self.data['cell_type'].unique())


raw_data_file_info = RawDataFileInfo(URL_FILE,RAW_DATA_DIR)