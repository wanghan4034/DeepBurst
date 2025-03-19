import numpy as np
import pandas as pd
import anndata as ad
import seaborn as sns 
import scanpy as sc 
from scipy.io import mmread
from scipy.stats import median_abs_deviation
from anndata import AnnData
from copy import deepcopy

sc.settings.verbosity = 0
sc.settings.set_figure_params(
    dpi=80,
    facecolor="white",
    frameon=False,
)


def sc_preprocess(data: AnnData, outlier_threshold = 3):
    adata = deepcopy(data)
    # mitochondrial genes
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # ribosomal genes
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    # hemoglobin genes.
    adata.var["hb"] = adata.var_names.str.contains(("^HB[^(P)]"))


    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo", "hb"], inplace=True, percent_top=[20], log1p=True
    )
    adata


    p1 = sns.displot(adata.obs["total_counts"], bins=100, kde=False)
    # sc.pl.violin(adata, 'total_counts')
    p2 = sc.pl.violin(adata, "pct_counts_mt")
    p3 = sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt",size=5)


    def is_outlier(adata, metric: str, nmads: int):
        M = adata.obs[metric]
        outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
            np.median(M) + nmads * median_abs_deviation(M) < M
        )
        return outlier



    adata.obs["outlier"] = (
        is_outlier(adata, "log1p_total_counts", outlier_threshold)
        | is_outlier(adata, "log1p_n_genes_by_counts", outlier_threshold)
        | is_outlier(adata, "pct_counts_in_top_20_genes", outlier_threshold)
    )
    adata.obs.outlier.value_counts()

    adata.obs["mt_outlier"] = is_outlier(adata, "pct_counts_mt", outlier_threshold) | (
        adata.obs["pct_counts_mt"] > 8
    )
    adata.obs.mt_outlier.value_counts()


    print(f"Total number of cells: {adata.n_obs}")
    adata = adata[(~adata.obs.outlier) & (~adata.obs.mt_outlier)].copy()

    print(f"Number of cells after filtering of low quality cells: {adata.n_obs}")

    p1 = sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt", size = 10)
    return adata


if __name__ == '__main__':
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
        return adata

    # counts =  pd.read_csv('extra/burst/raw_data/HepG2/HepG2_RNA.txt', sep='\t',index_col=0).T 
    # adata = ad.AnnData(counts)

    adata = load_data('extra/burst/raw_data/gm12878',genes_type="genes")
    adata.var_names_make_unique()
    processed = sc_preprocess(adata)