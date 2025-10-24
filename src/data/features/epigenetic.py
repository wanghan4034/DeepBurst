import os
import pyBigWig
import re
from functools import reduce
from typing import List, Dict, Union
import numpy as np
from src.data.features.basic import GenomicSlice, GenomicSite
from src.utils.logs import get_logger
logger = get_logger()




#%% epigenetic infomation
# cell type, epigenetic type, location ,expression

def replace_outlier(x: np.ndarray)->np.ndarray:
    median = np.percentile(x, 50)
    x[x > 1000] = median
    return np.log1p(x,dtype=np.float32)
 

class BigWigInfo:
    def __init__(self, file_paths:Union[str,List[str]]):
        self.data = self._read_all_file_info(file_paths)

    def _read_file_info(self,file_path):
        bw = pyBigWig.open(file_path)
        return bw
    
    def _read_all_file_info(self, file_paths):
        if not isinstance(file_paths,List):
            file_paths = [file_paths]

        data =  [self._read_file_info(file_path) for file_path in file_paths]
        return data
    
    def _get_value_from_bigwig(self, genomic_slice: 'Union[str,GenomicSlice]')->np.ndarray:
        if isinstance(genomic_slice,str):
            chrom = genomic_slice.split(':')[0]
            start = int(genomic_slice.split(':')[1].split('-')[0])
            end = int(genomic_slice.split(':')[1].split('-')[1])
        else:
            chrom = genomic_slice.chrom
            start = int(genomic_slice.start.location)
            end = int(genomic_slice.end.location)
        value = np.zeros(end-start, dtype=np.float32)
        if self.data:
            value = reduce(lambda x,y: x+y,[np.float32(np.nan_to_num(bw.values(chrom, start, end,numpy=True))) for bw in self.data])/len(self.data)

        return value
    
    def get_values_by_location(self, genomic_slices: 'Union[GenomicSlice,List[GenomicSlice]]')->'List[np.ndarray]':
        results = []
        if not isinstance(genomic_slices,List):
            genomic_slices = [genomic_slices]

        for genomic_slice in genomic_slices:
            results.append(self._get_value_from_bigwig(genomic_slice))
        return results

    def _get_stats_value_from_bigwig(self, location: 'GenomicSlice', statistic:str = 'mean')->np.float32:
        chrom = location.chrom
        start = location.start.location
        end = location.end.location
        value = np.float32(0.0)
        if self.data:
            value = (reduce(lambda x,y: x+y,[replace_outlier(np.nan_to_num(bw.values(chrom, start,end,numpy=True))) for bw in self.data])/len(self.data))
            if statistic == 'mean':
                value = np.mean(value)
            
            if statistic == 'max':
                value = np.max(value)
            
            if statistic == 'median':
                value = np.percentile(value, q=50)
            
            if statistic == '75_percentile':
                value = np.percentile(value,75)
        return value

    def get_stats_value_by_location(self, locations: List[GenomicSlice],statistic:str = 'mean')->List[np.float32]:
        results = []
        for location in locations:
            results.append(self._get_stats_value_from_bigwig(location, statistic))

        return results

if __name__ == '__main__':
    from src.data.features.genomics import get_genomic_slice
    file_paths = ['extra/datasets/epigenetic/E003-H3K4me1.bw']
    bigwig_info = BigWigInfo(file_paths)
    genomic_slice = get_genomic_slice('chr6',37321757,37321758)