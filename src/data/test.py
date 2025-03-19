import os
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from src.data.genomics import get_genomic_slice, get_dna_sequences
from src.data.utils import get_base_distribution
from src.data.epigenetic import BigWigInfo
from src.utils.logs import get_logger
logger = get_logger()

eid = 'E118'
marks = ['H3K4me1', 'H3K4me3', 'H3K9me3', 'H3K27me3', 'H3K36me3', 'H3K27ac', 'H3K9ac']
# marks = ['H3K4me1', 'H3K4me3']

chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']
meta_data_path = 'extra/datasets/meta_datasets/meta_data_E118.csv'
save_dir = 'extra/datasets/processed/E118'


train = pd.read_csv(meta_data_path)
train = train[train.eid == eid].reset_index(drop=True)

region_set = set()
for r in train.to_records():
    # Add TSS to region set.
    genomic_slice = get_genomic_slice(chrom=r['chrom'], start=r['start'] - 20000, end=r['start'] + 20000,strand=r['strand'])
    region_set.add(genomic_slice)

    # Add neighbors to region set.
    if  not pd.isnull(r['neighbors']):
        for neighbor in r['neighbors'].split(';'):
            chrom = neighbor.split(':')[0]
            start =  int(neighbor.split(':')[1].split('-')[0])
            end =  int(neighbor.split(':')[1].split('-')[1])
            genomic_slice = get_genomic_slice(chrom=chrom, start=start, end=end, strand=r['strand'])
            region_set.add(genomic_slice)

print('Loading genomewide read depth signals...')
s = time.time()
signal = {}

for mark in marks:
    bigwig_file_path = os.path.join('extra/datasets/epigenetic',f'{eid}-{mark}.bw')
    signal[mark] = BigWigInfo(bigwig_file_path)


if not os.path.exists(save_dir):
    os.makedirs(save_dir)

sequences = get_dna_sequences(list(region_set))

out = os.path.join(save_dir, f'{eid}.pkl')

samples = {}
for mark in marks:
    logger.info(f"Extracting {eid} {mark} signal")
    samples[mark] = signal[mark].get_values_by_location(region_set)




samples['seq'] = sequences

with open(out, 'wb') as f:
    pickle.dump(samples, f)