import os
from pathlib import Path

import scanpy as sc
import pandas as pd
import numpy as np

from scipy.io import mmread
from joblib import delayed, Parallel
from tqdm import tqdm

from src.data_preprocessing.label_generation.txburst.preprocess import sc_preprocess
from src.data_preprocessing.label_generation.txburst.txburstML import MaximumLikelihood, whichKeep
from src.utils.logs import get_logger

logger = get_logger()

# -----------------------------
# Constants and paths
# -----------------------------

EXPRESSION = "expression"
BF = "bf"
BS = "bs"
K_ON = "k_on"
K_OFF = "k_off"
K_SYN = "k_syn"
KINETICS = "kinetics"
KEEP = "keep"

NJOBS = 50
TARGET_CELL_CYCLE_PHASE = "S"
PHASE_ORIGIN = 'deepcycle'

DATA_DIR = "extra/datasets/burst/raw_data/gm12878"
CELL_QC_PATH = f"extra/datasets/burst/raw_data/gm12878/cellQC_{PHASE_ORIGIN}.tsv"

SAVE_DIR = "extra/datasets/burst/cellcycle_txburst/gm12878"
os.makedirs(SAVE_DIR, exist_ok=True)

print("DATA_DIR:", DATA_DIR)
print("CELL_QC_PATH:", CELL_QC_PATH)
print("SAVE_DIR:", SAVE_DIR)


def load_data(data_dir, genes_type="features", preprocess=False):
    """
    Load GM12878 single-cell expression data.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing 10x matrix files.
    genes_type : {"features", "genes"}
        - "features": use features.tsv.gz via scanpy.read_10x_mtx
        - "genes": manually read genes.tsv.gz
    preprocess : bool
        Whether to apply sc_preprocess. Default False to preserve raw counts.

    Returns
    -------
    adata : AnnData
        Cell x gene AnnData object.
    """
    data_dir = Path(data_dir)

    if genes_type == "features":
        adata = sc.read_10x_mtx(
            data_dir,
            var_names="gene_symbols",
            cache=True
        )
        adata.var_names_make_unique()

    else:
        matrix_path = data_dir / "matrix.mtx.gz"
        genes_path = data_dir / "genes.tsv.gz"
        barcodes_path = data_dir / "barcodes.tsv.gz"

        matrix = mmread(matrix_path).T.tocsr()
        genes = pd.read_csv(genes_path, header=None, sep="\t")
        barcodes = pd.read_csv(barcodes_path, header=None, sep="\t")

        adata = sc.AnnData(X=matrix)
        adata.var_names = genes[1].astype(str).values
        adata.var["gene_ids"] = genes[0].astype(str).values
        adata.obs_names = barcodes[0].astype(str).values
        adata.var_names_make_unique()

    # Preserve raw counts for txburst
    adata.layers["counts"] = adata.X.copy()

    if preprocess:
        adata = sc_preprocess(adata)
        if "counts" not in adata.layers:
            adata.layers["counts"] = adata.X.copy()

    print(adata)
    return adata

adata = load_data(
    DATA_DIR,
    genes_type="features",
    preprocess=True
)

adata.obs.head()

cell_qc = pd.read_csv(
    CELL_QC_PATH,
    sep="\t",
    index_col=0
)

cell_qc.index = cell_qc.index.astype(str)

print(cell_qc.shape)
cell_qc.head()


# adata.obs_names: AAACCTGAGACCCACC-1
# cell_qc.index:   AAACCTGAGACCCACC
# Therefore, remove trailing '-1' from adata.obs_names before matching.

adata.obs["barcode"] = adata.obs_names.astype(str)
adata.obs["barcode_no_suffix"] = adata.obs["barcode"].str.replace(
    r"-1$",
    "",
    regex=True
)

adata.obs = adata.obs.merge(
    cell_qc,
    left_on="barcode_no_suffix",
    right_index=True,
    how="left"
)

print("adata cells:", adata.n_obs)
print("matched cellQC:", adata.obs["cellPhase"].notna().sum())
print("unmatched:", adata.obs["cellPhase"].isna().sum())

adata.obs.head()

adata.obs["phase"] = (
    adata.obs["cellPhase"]
    .astype(str)
    .str.upper()
    .str.replace("/", "", regex=False)
    .str.replace("-", "", regex=False)
    .str.replace("_", "", regex=False)
)

adata.obs["phase"] = adata.obs["phase"].replace({
    "G0G1": "G1",
    "G1": "G1",
    "S": "S",
    "G2M": "G2M",
    "G2": "G2M",
    "M": "G2M",
    "NAN": pd.NA,
})

adata.obs["phase"].value_counts(dropna=False)

adata_g1 = adata[adata.obs["phase"] == TARGET_CELL_CYCLE_PHASE].copy()

def adata_to_gene_by_cell_df(adata_input, use_layer="counts"):
    """
    Convert AnnData to a gene x cell count DataFrame.

    txburst inference below expects:
        rows = genes
        columns = cells
    """
    if use_layer is not None and use_layer in adata_input.layers:
        X = adata_input.layers[use_layer]
    else:
        X = adata_input.X

    if hasattr(X, "toarray"):
        X = X.toarray()

    data = pd.DataFrame(
        X.T,
        index=adata_input.var_names.astype(str),
        columns=adata_input.obs_names.astype(str)
    )

    # Merge duplicated gene symbols if any
    if data.index.duplicated().any():
        data = data.groupby(data.index).sum()

    return data

# No low-expression gene filtering is applied.
data_g1 = adata_to_gene_by_cell_df(
    adata_g1,
    use_layer="counts"
)

print(data_g1.shape)  # expected: genes x G1 cells
data_g1.iloc[:5, :5]


def calculate_stats(data):
    """
    Calculate mean expression for each gene.

    Input
    -----
    data : pd.DataFrame
        gene x cell count matrix.
    """
    print("calculate_stats dataframe shape:", data.shape)

    data_stats = pd.DataFrame(index=data.index)
    data_stats[EXPRESSION] = data.mean(axis=1)

    return data_stats[[EXPRESSION]]


data_stats_g1 = calculate_stats(data_g1)
data_stats_g1.head()


def compute_cell_size_ratio(data):
    """
    Compute cell-size ratio for txburst.

    Input
    -----
    data : pd.DataFrame
        gene x cell count matrix.

    Returns
    -------
    np.ndarray
        cell-size ratio with length = n_cells.
    """
    T = data.sum(axis=0)
    T_bar = T.mean()

    if T_bar <= 0:
        return np.ones(data.shape[1])

    cell_size_ratio = T / T_bar
    cell_size_ratio = np.clip(cell_size_ratio, 0.01, 10)

    return cell_size_ratio.values


def infer_one_gene_txburst(rpkm, delay=1, cell_size_ratio=1):
    """
    Run txburst for one gene.

    rpkm : pd.Series
        One row of data_g1, i.e. one gene across all G1 cells.
    """
    try:
        x = np.asarray(rpkm[pd.notnull(rpkm)], dtype=float)
        x = np.around(x)

        param = MaximumLikelihood(
            x,
            "L-BFGS-B",
            delay,
            cell_size_ratio
        )

        return param

    except Exception:
        return None


def infer_kinetics(data: pd.DataFrame, delay: float = 1, with_cell_size: bool = False,n_jobs: int = 1) -> pd.DataFrame:
    print('infer_kInferring kinetics dataframe shape:', data.shape)
    
    cell_size_ratio = compute_cell_size_ratio(data) if with_cell_size else 1
    params = Parallel(n_jobs=n_jobs, verbose = 3)(delayed(MaximumLikelihood)(np.around(rpkm[pd.notnull(rpkm)]),'L-BFGS-B',delay, cell_size_ratio) for _,rpkm in tqdm(data.iterrows(),total=len(data)))
    keep = whichKeep(params)

    print('Inferred kinetics of {} genes out of {} total'.format(np.sum(keep), len(keep)))   
    kinetics = pd.DataFrame([ params, list(keep)], columns=data.index).T
    kinetics.columns = [KINETICS, KEEP]
    kinetics[K_ON] = kinetics.apply(lambda row: row[KINETICS][0], axis = 1)
    kinetics[K_OFF] = kinetics.apply(lambda row: row[KINETICS][1], axis = 1)    
    kinetics[K_SYN] = kinetics.apply(lambda row: row[KINETICS][2], axis = 1)
    return kinetics[[K_ON,K_OFF,K_SYN]]


delay = 1
with_cell_size = True

data_kinetics_g1 = infer_kinetics(
    data_g1,
    delay=delay,
    with_cell_size=with_cell_size,
    n_jobs=NJOBS
)

data_kinetics_g1.head()


g1_result = pd.concat(
    [data_stats_g1, data_kinetics_g1],
    axis=1
)

g1_result["gene_name"] = g1_result.index

g1_result.head()
g1_result.fillna(0.0).to_csv(os.path.join(SAVE_DIR,f'gm12878_statistic_gene_transcript_region_delay_{delay}_with_cellsize_{int(with_cell_size)}_cellcycle_{TARGET_CELL_CYCLE_PHASE}_{PHASE_ORIGIN}.csv'), index = False)