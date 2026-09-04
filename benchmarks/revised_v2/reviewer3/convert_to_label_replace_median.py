import os
import pandas as pd 
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--eid', required=True, help='Cell Type EID.')
parser.add_argument('--delay', required=False, help='Capture efficiency delay.',default=1,type=float)
parser.add_argument("--with_cell_size",action="store_true",help="Enable cell size ratio correction.")
parser.add_argument('--gene_id2neighbors', required=True, help='gene_id2neighbors data file.')
parser.add_argument('--processed_dir', required=False, help='processed_dir',default='extra/datasets/burst/processed')
parser.add_argument('-o','--output', required=True, help='Input tissue.')

args = parser.parse_args()

eid = args.eid
delay = args.delay
with_cell_size = args.with_cell_size
gene_id2neighbors = args.gene_id2neighbors
processed_dir = args.processed_dir
saved_dir = args.output


EIDS_TO_CELLTYPES = {
    'E116':'gm12878',
    'E118':'HepG2',
    'E003':'H1'
}

# TARGET_INDICATOR = 'cv'

if not os.path.exists(saved_dir):
    os.makedirs(saved_dir)

kinetics = pd.read_csv(os.path.join(processed_dir,f'{EIDS_TO_CELLTYPES[eid]}_statistic_gene_transcript_region_delay_{delay}_with_cellsize_{int(with_cell_size)}.csv'))
bulk_exp = pd.read_csv('extra/datasets/annotations/exp_raw.tsv',sep='\t')[['gene_id',eid]]
bulk_exp.columns = ['gene_id','bulk_exp']
gene_id2neighbors = pd.read_csv(gene_id2neighbors, names=['gene_id','eid','chrom','start','end','strand','neighbors','scores'])

meta_data = pd.merge(gene_id2neighbors,kinetics, how='inner', on='gene_id')
meta_data = pd.merge(meta_data,bulk_exp, how='inner', on='gene_id')


# BS=mu/beta；
# BF=(alpha*beta)/(alpha+beta)
# meta_data['expression'] = meta_data['mean']
meta_data['bs'] = meta_data['k_syn']/meta_data['k_off'] # ksyn/koff
meta_data['bf'] = meta_data['k_on']*meta_data['k_off']/(meta_data['k_on']+meta_data['k_off']) # (kon*koff)/(kon+koff)
meta_data['mean'] = meta_data['k_on']*meta_data['k_syn']/(meta_data['k_on']+meta_data['k_off'])
meta_data['cv'] = meta_data.apply(lambda row: 0 if row['mean'] < 0.0001 else 1/row['mean'] + row['k_off']/row['k_on'] - ((row['k_on'] + row['k_off'])/row['k_on'] )*(1-(row['k_on']/(1+row['k_on'])))/((1+row['k_off'])/row['k_off'] - row['k_on']/(1+row['k_on'])),axis=1)


bulk_exp_median =  np.median(meta_data['bulk_exp'])
sc_exp_median = np.median(meta_data['expression'])

meta_data['bulk_exp_label'] = meta_data['bulk_exp'] > bulk_exp_median
meta_data['bulk_exp_label'] = np.int32(meta_data['bulk_exp_label'])

meta_data['sc_exp_label'] = meta_data['expression'] > sc_exp_median
meta_data['sc_exp_label'] = np.int32(meta_data['sc_exp_label'])

meta_data = meta_data[meta_data['sc_exp_label'] == meta_data['bulk_exp_label']]

# 使用mean 作为阈值
bs_mean = np.mean(meta_data['bs'])
bf_mean = np.mean(meta_data['bf'])
cv_mean = np.mean(meta_data['cv'])
mean_mean = np.mean(meta_data['mean'])

meta_data['bs_label'] = meta_data.apply(lambda row: 1 if row['bs'] > bs_mean else 0, axis=1)
meta_data['bf_label'] = meta_data.apply(lambda row: 1 if row['bf'] > bf_mean else 0, axis=1)
meta_data['cv_label'] = meta_data.apply(lambda row: 1 if row['cv'] > cv_mean else 0, axis=1)
meta_data['mean_label'] = meta_data.apply(lambda row: 1 if row['mean'] > mean_mean else 0, axis=1)

bs_threshold = np.percentile(meta_data['bs'],95)
bf_threshold = np.percentile(meta_data['bf'],95)
mean_threshold = 1

meta_data = meta_data[((meta_data['bs'] < bs_threshold) & (meta_data['bf'] < bf_threshold)) | (meta_data['mean'] > mean_threshold)]
meta_data.to_csv(os.path.join(saved_dir,f'meta_data_{eid}_delay_{delay}_with_cellsize_{int(with_cell_size)}_threshold_mean.csv'),index=False)
