import os
import scanpy as sc
import pandas as pd
import numpy as np
import argparse 
from scipy.io import mmread
from joblib import delayed,Parallel
from tqdm import tqdm
from src.data_preprocessing.label_generation.txburst.preprocess import sc_preprocess
from src.data_preprocessing.label_generation.txburst.txburstML import MaximumLikelihood, whichKeep

from src.utils.logs import get_logger
logger = get_logger()


EXPRESSION = 'expression'
CV = 'cv'
VARIANCE = 'variance'
BF = 'bf'
BS = 'bs'
K_ON = 'k_on'
K_OFF = 'k_off'
K_SYN = 'k_syn'
KINETICS = 'kinetics'
KEEP = 'keep'
NJOBS = 50
DELAY = 1

def load_data(data_dir, genes_type="features"):
    """
    Load data either from 10x Genomics formatted data (features) or from specified paths (matrix, genes, and barcodes).
    """
    if genes_type == "features":
        adata = sc.read_10x_mtx(data_dir, var_names='gene_symbols', cache=True)
    else:
        matrix_path = "{}/matrix.mtx.gz".format(data_dir)
        genes_path = "{}/genes.tsv.gz".format(data_dir)
        barcodes_path = "{}/barcodes.tsv.gz".format(data_dir)
        # Read matrix, genes, and barcodes files
        matrix = mmread(matrix_path).T.tocsr()
        genes = pd.read_csv(genes_path, header=None, sep='\t')
        barcodes = pd.read_csv(barcodes_path, header=None, sep='\t')
        adata = sc.AnnData(X=matrix)
        adata.var_names = genes[1].values
        adata.var['gene_ids'] = genes[0].values
        adata.obs_names = barcodes[0].values
        adata.var_names_make_unique()
        adata = sc_preprocess(adata)
    return adata


def calculate_stats(data:pd.DataFrame):
    """
    Calculate the mean, variance, and coefficient of variation (CV) for the given data, and summarize the results into a dataframe.
    """
    data_ = data.copy()

    columns = data_.columns
    data_[EXPRESSION] = data_[columns].apply(lambda row:np.mean(row), axis=1)
    data_[VARIANCE] = data_[columns].apply(lambda row:np.var(row), axis=1)
    data_[CV] = data_[columns].apply(lambda row:np.std(row)/np.mean(row), axis=1)
    return data_[[EXPRESSION, VARIANCE, CV]]


def load_gene_name_code(data_dir, genes_type="features", gene_columns=["gene_id", "gene_name"]):
    """
    Load gene name and code information from a specified file.
    """
    genes_path = "{}/{}.tsv".format(data_dir, genes_type)
    genes = pd.read_csv(genes_path, header=None, sep='\t').iloc[:, :2]
    genes.columns = gene_columns
    return genes


def read_data(root_data_path, cell_type):

    if cell_type == "K562":
        data_dir = os.path.join(root_data_path, "K562")
        genes_type = "genes"
    elif cell_type == "gm12878":
        data_dir = os.path.join(root_data_path , "gm12878")
        genes_type = "genes"
    elif cell_type == "HepG2":
        data_dir = os.path.join(root_data_path, "HepG2")
        genes_type = "features"
    elif cell_type == "keratinocytes":
        data_dir = os.path.join(root_data_path, "keratinocytes/GSE155850")
        genes_type = "features"
    elif cell_type == "H1":
        data_dir = os.path.join(root_data_path, "H1")
        genes_type = "features"
    elif cell_type == "H1_primed":
        data_dir = os.path.join(root_data_path, "H1_primed")
        genes_type = "features"
    elif cell_type == "OPC_human":
        data_dir = os.path.join(root_data_path, "OPC_human")
        genes_type = "features"
    elif cell_type == "OPC_macaque":
        data_dir = os.path.join(root_data_path, "OPC_macaque")
        genes_type = "features"
    elif cell_type == "OPC_mouse":
        data_dir = os.path.join(root_data_path, "OPC_mouse")
        genes_type = "features"
    if cell_type == "HepG2":
        data_df = pd.read_csv(os.path.join(data_dir ,"HepG2_RNA.txt"), sep='\t').T
    elif cell_type == "OPC_human":
        data_df = pd.read_csv(os.path.join(data_dir ,"OPC_human.csv"), sep=',')
    elif cell_type == "OPC_macaque":
        data_df = pd.read_csv(os.path.join(data_dir ,"OPC_macaque.csv"), sep=',')
    elif cell_type == "OPC_mouse":
        data_df = pd.read_csv(os.path.join(data_dir ,"OPC_mouse.csv"), sep=',')
    elif cell_type == "H1":
        data_df = pd.read_csv(os.path.join(data_dir ,"H1_RNA.csv"), sep=',',index_col=0).T
    elif cell_type == "H1_primed":
        data_df = pd.read_csv(os.path.join(data_dir ,"H1_primed_RNA.csv"), sep=',',index_col=0).T
    elif cell_type == "gm12878":
        adata = load_data(data_dir, genes_type=genes_type)
        data_df = pd.DataFrame(adata.X.toarray(), columns=adata.var_names, index=adata.obs_names)
    else:
        raise ValueError(f"No target cell type:{cell_type}")
    

    
    print(data_df.head())

    return data_df.T


def compute_cell_size_ratio(data: pd.DataFrame):
    T = data.sum(axis=0)          # T_j, 对每个细胞列求和
    T_bar = T.mean()            # \bar T
    cell_size_ratio = T / T_bar        
    cell_size_ratio = np.clip(cell_size_ratio, 0.01, 10)   # T_j / \bar T
    return cell_size_ratio.values

def infer_kinetics(data: pd.DataFrame):
    print('Inferring kinetics:')
    cell_size_ratio = compute_cell_size_ratio(data)
    params = Parallel(n_jobs=NJOBS, verbose = 3)(delayed(MaximumLikelihood)(np.around(rpkm[pd.notnull(rpkm)]),'L-BFGS-B',DELAY, cell_size_ratio) for _,rpkm in tqdm(data.iterrows(),total=len(data)))
    keep = whichKeep(params)

    print('Inferred kinetics of {} genes out of {} total'.format(np.sum(keep), len(keep)))   
    kinetics = pd.DataFrame([ params, list(keep)], columns=data.index).T
    kinetics.columns = [KINETICS, KEEP]
    kinetics[K_ON] = kinetics.apply(lambda row: row[KINETICS][0], axis = 1)
    kinetics[K_OFF] = kinetics.apply(lambda row: row[KINETICS][1], axis = 1)    
    kinetics[K_SYN] = kinetics.apply(lambda row: row[KINETICS][2], axis = 1)
    kinetics[BF] = kinetics[K_ON]
    kinetics[BS] = kinetics.apply(lambda row: row[KINETICS][2]/row[KINETICS][1], axis = 1)
    return kinetics[[BS,BF,K_ON,K_OFF,K_SYN]]


def main(cell_type, data_dir, saved_dir):

    logger.info(f"burst kinetics inference, celltype: {cell_type}")

    data = read_data(data_dir, cell_type=cell_type)

    data_stats = calculate_stats(data)
    data_kinetics = infer_kinetics(data)

    merge_stats = pd.concat([data_stats,data_kinetics], axis = 1)
    merge_stats['gene_name'] = merge_stats.index
    gene_name_code = load_gene_name_code(os.path.join(data_dir,cell_type))
    merged_data = pd.merge(merge_stats, gene_name_code, on="gene_name", how='inner')
    merged_data.fillna(0.0).to_csv(os.path.join(saved_dir,f'{cell_type}_statistic_gene_transcript_region_with_cell_size.csv'), index = False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--cell_type', required=True, help='cell type.')
    parser.add_argument('--data_dir', required=False, help='raw_data.',default='extra/datasets/burst/raw_data/')
    parser.add_argument('-o','--output', required=False, help='Input tissue.',default='extra/datasets/burst/processed')

    args = parser.parse_args()

    cell_type = args.cell_type
    data_dir = args.data_dir
    saved_dir = args.output 
    main(cell_type,data_dir,saved_dir)