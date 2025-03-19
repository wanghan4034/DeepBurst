import os 
import yaml
import numpy as np 
from typing import Dict, List

HG19 = 'hg19'
HG38 = 'hg38'
COORD = 'hg19'

### epigenetic data

# Assays
TOTAL_RNA_SEQ = 'total RNA-seq'
POLYA_PLUS_RNA_SEQ = 'polyA plus RNA-seq'
HISTONE_CHIP_SEQ = 'Histone ChIP-seq'
TF_CHIP_SEQ = 'TF ChIP-seq'
DNASE_SEQ = 'DNase-seq'

# experiment targets
CTCF_HUMAN = 'CTCF-human'
H3K4ME3_HUMAN = 'H3K4me3-human'
H3K4ME1_HUMAN = 'H3K4me1-human'
H3K27AC_HUMAN = 'H3K27ac-human'
H3K4ME2_HUMAN = 'H3K4me2-human'
H3K36ME3_HUMAN = 'H3K36me3-human'
H3K27ME3_HUMAN = 'H3K27me3-human'
H3K79ME2_HUMAN = 'H3K79me2-human'
H3K9AC_HUMAN = 'H3K9ac-human'
H3K9ME3_HUMAN = 'H3K9me3-human'
H4K20ME1_HUMAN = 'H4K20me1-human'
H2AFZ_HUMAN = 'H2AFZ-human'
EP300_HUMAN = 'EP300-human'
POLR2A_HUMAN = 'POLR2A-human'
POLR2A_PHOSPHOS5_HUMAN = 'POLR2AphosphoS5-human'


HISTONE_CHIP_EXPERIMENT_TARGETS = [H3K4ME3_HUMAN,H3K4ME1_HUMAN,H3K27AC_HUMAN,H3K4ME2_HUMAN,H3K36ME3_HUMAN,H3K27ME3_HUMAN,H3K79ME2_HUMAN,H3K9AC_HUMAN,H3K9ME3_HUMAN,H4K20ME1_HUMAN,H2AFZ_HUMAN]
TF_CHIP_EXPERIMENT_TARGETS = [CTCF_HUMAN,POLR2A_PHOSPHOS5_HUMAN,POLR2A_HUMAN,EP300_HUMAN]


EXPRESSION_ASSAY = [TOTAL_RNA_SEQ,POLYA_PLUS_RNA_SEQ]
EPIGENETIC_ASSAY = [HISTONE_CHIP_SEQ,TF_CHIP_SEQ,DNASE_SEQ]


CRES_PATH = 'extra/datasets/CREs/GRCh38-cCREs.bed'
GENOMIC_ANNOTATION_PATH = 'extra/datasets/CREs/gencode.v46.annotation.gtf.gz'


# expression
# CREs
# Epigenetic modification data
# CHIA-PET



#%% raw data file process


TOTAL_ASSAYS = EXPRESSION_ASSAY + EPIGENETIC_ASSAY
EXPERIMENT_TARGETS = HISTONE_CHIP_EXPERIMENT_TARGETS + TF_CHIP_EXPERIMENT_TARGETS + [DNASE_SEQ]

TARGET_CELL_TYPES = ['GM12878','HepG2','H1']

ASSAY_EXPERIMENT_TARGETS_MAP = {
    TOTAL_RNA_SEQ:[],
    POLYA_PLUS_RNA_SEQ:[],
    TF_CHIP_SEQ:TF_CHIP_EXPERIMENT_TARGETS,
    DNASE_SEQ:[],
    HISTONE_CHIP_SEQ:HISTONE_CHIP_EXPERIMENT_TARGETS,
}


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

CONFIG_PATH = 'src/model/configs/default.yaml'
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