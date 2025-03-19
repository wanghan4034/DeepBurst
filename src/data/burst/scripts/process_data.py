import scanpy as sc
import pandas as pd
import numpy as np
from scipy.io import mmread

def load_data(data_dir, genes_type="features"):
    """
    Load data either from 10x Genomics formatted data (features) or from specified paths (matrix, genes, and barcodes).
    """
    if genes_type == "features":
        adata = sc.read_10x_mtx(data_dir, var_names='gene_symbols', cache=True)
    else:
        matrix_path = "{}/matrix.mtx.gz".format(data_dir)
        genes_path = "{}/genes.tsv".format(data_dir)
        barcodes_path = "{}/barcodes.tsv.gz".format(data_dir)
        # Read matrix, genes, and barcodes files
        matrix = mmread(matrix_path).T.tocsr()
        genes = pd.read_csv(genes_path, header=None, sep='\t')
        barcodes = pd.read_csv(barcodes_path, header=None, sep='\t')
        adata = sc.AnnData(X=matrix)
        adata.var_names = genes[1].values
        adata.var['gene_ids'] = genes[0].values
        adata.obs_names = barcodes[0].values
    return adata

def filter_data_by_counts(adata, threshold=5000):
    """
    Calculate the number of genes expressed per cell and filter out cells with a number of expressed genes less than or equal to the threshold.
    """
    adata.obs['n_genes_by_counts'] = (adata.X > 0).sum(axis=1)
    sc.pp.filter_cells(adata, min_genes=threshold + 1)
    return adata

def calculate_stats(data:pd.DataFrame):
    """
    Calculate the mean, variance, and coefficient of variation (CV) for the given data, and summarize the results into a dataframe.
    """
    mean = data.mean()
    variance = data.var()
    cv = data.std() / mean 
    cv = cv.fillna(0.0)
    summary_df = pd.DataFrame({
        'mean': mean,
        'variance': variance,
        'CV': cv
    })
    return summary_df

def load_gene_name_code(data_dir, genes_type="features", gene_columns=["gene_id", "gene_name"]):
    """
    Load gene name and code information from a specified file.
    """
    genes_path = "{}/{}.tsv".format(data_dir, genes_type)
    genes = pd.read_csv(genes_path, header=None, sep='\t').iloc[:, :2]
    genes.columns = gene_columns
    return genes

# Define paths and thresholds
root_data_path = "/GPUFS/sysu_jjzhang_3/wanghan/burst-chromoformer/preprocessing/burst/raw_data/"
processed_data_path = "/GPUFS/sysu_jjzhang_3/wanghan/burst-chromoformer/preprocessing/burst/processed"
cell_type = "gm12878"
mean_threshold = 0.001  # Filter threshold for the mean expression of single cells
count_threshold = 5000  # Threshold for the number of genes expressed in a cell
if cell_type == "k562":
    data_dir = root_data_path + "k562/GSM5687481"
    genes_type = "features"
elif cell_type == "gm12878":
    data_dir = root_data_path + "gm12878"
    genes_type = "genes"
elif cell_type == "HepG2":
    data_dir = root_data_path + "HepG2"
    genes_type = "features"
elif cell_type == "keratinocytes":
    data_dir = root_data_path + "keratinocytes/GSE155850"
    genes_type = "features"
    
# Load gene transcript region information
# gene_transcript_region_path = "{}/gene_transcript_region.txt".format(root_data_path)
# gene_transcript_region = pd.read_csv(gene_transcript_region_path, header=None, sep='\t')
# gene_transcript_region.columns = ["gene_code", "chr", "start", "end"]
# gene_transcript_region = gene_transcript_region.drop_duplicates(subset=['gene_code'], keep='first')

# Load and preprocess the data
if cell_type == "HepG2":
   data_df = pd.read_csv(data_dir + "/HepG2_RNA.txt", sep='\t').T
   non_zero_counts = (data_df != 0).sum(axis=1)
   filtered_adata_df = data_df.loc[non_zero_counts >= count_threshold,:]
   print(data_df.head())
else: 
    adata = load_data(data_dir, genes_type=genes_type)
    # filtered_adata = filter_data_by_counts(adata, count_threshold)

    # Convert AnnData to DataFrame
    filtered_adata_df = pd.DataFrame(adata.X.toarray(), columns=adata.var_names, index=adata.obs_names)



# filtered_adata_df = filtered_adata_df.T

# filtered_adata_df['mean'] = filtered_adata_df.apply(lambda row: np.mean(row), axis=1)

# filtered_adata_df = filtered_adata_df[filtered_adata_df['mean'] > mean_threshold]

# Compute statistics for the filtered data
data_stats = calculate_stats(filtered_adata_df)
filtered_data_stats = data_stats[data_stats["mean"] > mean_threshold].reset_index()
filtered_genes = list(filtered_data_stats['index'])
data_for_kinetic = filtered_adata_df[filtered_genes].T.reset_index()

print(filtered_data_stats.shape)
print(filtered_data_stats.head())

# Load gene name and code information
gene_name_code = load_gene_name_code(data_dir, genes_type=genes_type)


# Merge statistical data with gene name and transcript region information and save the results to a CSV file
merged_data = pd.merge(filtered_data_stats, gene_name_code, left_on='index', right_on="gene_name", how='left')
# merged_data2 = pd.merge(merged_data, gene_transcript_region, left_on='gene_code', right_on="gene_code", how='inner').drop(columns=['index'])
merged_data_for_kinetic = pd.merge(data_for_kinetic, gene_name_code, left_on='index', right_on="gene_name", how='inner')
merged_data_for_kinetic.index = list(merged_data_for_kinetic['gene_id'])
merged_data_for_kinetic.drop(columns=['index','gene_id','gene_name']).dropna().to_csv(f'{processed_data_path}/{cell_type}_UMI_filter_counts.csv')
merged_data.drop(columns=['index']).dropna().to_csv("{}/{}_statistic_gene_transcript_region.csv".format(processed_data_path, cell_type), index=False)
