import os 
import yaml
import numpy as np 
from typing import Dict, List

HG19 = 'hg19'
HG38 = 'hg38'
COORD = 'hg19'

### epigenetic data

#### data processing constants
CHROMOSOMES = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']

GENOMIC_FASTA_PATH = 'extra/datasets/genomic/hg19/hg19.fa'
FA_SIZE_PATH = 'extra/datasets/genomic/hg19/hg19.chrom.sizes'
with open(FA_SIZE_PATH,'r') as f:
    FASTA_SIZE = {line.strip().split('\t')[0]:int(line.strip().split('\t')[1]) for line in f.readlines()}

SCALED_WINDOW_SIZE = 20000
GENOMIC_SLICE = 'genomic_slice'
START_PADDING_SIZE = 'start_padding_size'
END_PADDING_SIZE = 'end_padding_size'

# CONFIG_PATH = 'src/model/configs/default.yaml'
CONFIG_PATH = 'configs/default.yaml'
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

MARKS = CONFIG['marks'].split('\t')
BASE_MAP = {l.split(':')[0]:int(l.split(':')[1]) for l in CONFIG['base_maps'].split('\t')}
CONFIG['marks'] = MARKS
CONFIG['base_map'] = BASE_MAP
CONFIG['marks_nums'] = len(MARKS)
CONFIG['base_nums'] = len(BASE_MAP) 


CONFIG['promoter_feats_basic_nums'] = CONFIG['marks_nums'] + CONFIG['base_nums'] if CONFIG['promoter_with_sequence'] else CONFIG['marks_nums']
CONFIG['pcres_feats_basic_nums'] = CONFIG['marks_nums'] + CONFIG['base_nums'] if CONFIG['pcres_with_sequence'] else CONFIG['marks_nums']