import os
from typing import List, Union
import numpy as np
import pandas as pd
from copy import deepcopy
from src.data.rawdata import FileInfo, raw_data_file_info
from src.data.constants import *


#%% Get expression 
class GeneExpressionInfo:
    def __init__(self, cell_types=TARGET_CELL_TYPES):
        file_infos = raw_data_file_info.get_file_index(cell_types=cell_types, assays=EXPRESSION_ASSAY, reduce_replicate=False)
        self.data = self._read_all_file_info(file_infos)
    
    def _read_file_info(self, file_info: FileInfo):
        """
        read a single file
        """

        if not os.path.exists(file_info.file_path):
            return pd.DataFrame()

        data = pd.read_csv(file_info.file_path, sep='\t')

        if 'gene_id' in data.columns:
            data = data[data['gene_id'].str.contains('ENSG')]
            data = data[['gene_id','TPM']]
        
        if 'target_id' in data.columns:
            data = data[['target_id','tpm']]
            data = data[data['target_id'].str.contains('ENSG')]
            data['target_id'] = data.apply(lambda row:row['target_id'].split('|')[1], axis=1)

        data.columns = ['gene_id','tpm']
        data = data[data['gene_id'].str.contains('ENSG')]
        data['gene_id'] = data.apply(lambda row:row['gene_id'].split('.')[0], axis=1)
        data = data.groupby(['gene_id'], as_index=False).sum()
        data['cell_type'] = file_info.cell_type
        return data
    
    def _read_all_file_info(self,file_infos:List[FileInfo]):
        """
        read all raw data file, expression equal log1e(tpm)
        """
        data = pd.concat([self._read_file_info(file_info) for file_info in file_infos])
        data = data.groupby(['cell_type','gene_id'], as_index=False).mean()
        data['tpm'] = data.apply(lambda row: np.log1p(row['tpm']), axis=1)
        return data

    def get_gene_expression(self, cell_types: Union[List[str], str] = [], target_genes: Union[List[str], str] = [])->pd.DataFrame:
        """
        target_genes = ['ENSG00000000003','ENSG00000285993']
        cell_types = ['GM23338','keratinocyte']
        """
        data = deepcopy(self.data)
        if isinstance(cell_types, str):
            cell_types = [cell_types]
        
        if isinstance(target_genes,str):
            target_genes = [target_genes]

        if cell_types:
            data = data[data['cell_type'].isin(cell_types)]
        
        if target_genes:
            data = data[data['gene_id'].isin(target_genes)]

        return data

    @property    
    def gene_expressions(self)->Dict[str,Dict[str,np.float32]]:
        data = deepcopy(self.data[['cell_type','gene_id','tpm']])
        results = {}
        for cell_type, gene_id, tpm in data.values:
            if cell_type not in results:
                results[cell_type] =  {}

            expressions = results[cell_type]
            expressions[gene_id] = tpm

        return results

gene_expression_info = GeneExpressionInfo(cell_types=TARGET_CELL_TYPES)