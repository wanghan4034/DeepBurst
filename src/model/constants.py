import os 
import yaml
import numpy as np 
from typing import Dict, List

HG19 = 'hg19'
HG38 = 'hg38'
COORD = 'hg19'

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

CONFIG_PATH = 'benchmark/Promoterformer/configs/default.yaml'
def get_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    marks = config['marks'].split('\t')
    base_map = {l.split(':')[0]:int(l.split(':')[1]) for l in config['base_maps'].split('\t')}
    config['marks'] = marks
    config['base_map'] = base_map
    config['marks_nums'] = len(marks)
    config['base_nums'] = len(base_map) 


    config['promoter_feats_basic_nums'] = config['marks_nums'] + config['base_nums'] if config['promoter_with_sequence'] else config['marks_nums']
    config['pcres_feats_basic_nums'] = config['marks_nums'] + config['base_nums'] if config['pcres_with_sequence'] else config['marks_nums']
    return config