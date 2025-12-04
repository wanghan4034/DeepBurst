# scripts/train_sklearn_bfbs.py

import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from src.model.constants import MARKS  # 你的 histone 名称列表，例如 ["H3K4me3", "H3K27ac", ...]

# ================== 特征提取 ================== #

def extract_histone_features_for_gene(
    npy_dir: str,
    eid: str,
    gene_id: str,
    binsizes: List[int] = [2000, 500, 100],
    use_marks: List[str] = None,
) -> np.ndarray:
    """
    从 {npy_dir}/{eid}/regions/{gene_id}.npy 读取组蛋白矩阵，
    对每个 binsize 做 mean-pooling + log1p，然后展平为一维特征。

    期望 .npy 形状: (n_feats, L_bp)
      - 前 len(MARKS) 行是 histone marks
    """
    npy_path = os.path.join(npy_dir, eid, "regions", f"{gene_id}.npy")
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"npy not found: {npy_path}")

    arr = np.load(npy_path)  # (n_feats, L_bp)
    n_feats, L = arr.shape

    if use_marks is None:
        use_marks = MARKS

    # 只取需要的 histone marks
    mark_idxes = [MARKS.index(m) for m in use_marks]
    x = arr[mark_idxes, :]  # (n_used_marks, L_bp)

    feat_list = []
    for bin_size in binsizes:
        n_bins = L // bin_size
        eff_len = n_bins * bin_size
        if eff_len == 0:
            # 极端情况：窗口太小
            feat_list.append(np.zeros((len(mark_idxes) * 1,), dtype=np.float32))
            continue

        x_eff = x[:, :eff_len]  # (n_marks, eff_len)
        x_bins = x_eff.reshape(len(mark_idxes), n_bins, bin_size).mean(axis=-1)  # (n_marks, n_bins)
        x_log = np.log1p(x_bins)  # log1p

        # 展平成一维: [mark1_bin1... mark1_binN, mark2_bin1...]
        feat_list.append(x_log.reshape(-1))

    features = np.concatenate(feat_list, axis=0)
    return features.astype(np.float32)


def build_feature_matrix(
    meta_path: str,
    npy_dir: str,
    binsizes: List[int] = [2000, 500, 100],
    bf_label_col: str = "bf_label",
    bs_label_col: str = "bs_label",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    从 meta.csv + npy 构建:
      X:       (N, D)
      y_bf:    (N,) -> bf_label
      y_bs:    (N,) -> bs_label
      chroms:  (N,) -> 每个 gene 的染色体
      gene_ids: List[str]
    """
    meta = pd.read_csv(meta_path)

    # 确保需要的列存在
    meta = meta.dropna(subset=[bf_label_col, bs_label_col, "gene_id", "eid", "chrom"])

    X_list = []
    y_bf_list = []
    y_bs_list = []
    chrom_list = []
    gene_ids = []

    for row in meta.itertuples(index=False):
        gene_id = getattr(row, "gene_id")
        eid = getattr(row, "eid")
        chrom = getattr(row, "chrom")
        bf_label = getattr(row, bf_label_col)
        bs_label = getattr(row, bs_label_col)

        try:
            feats = extract_histone_features_for_gene(
                npy_dir=npy_dir,
                eid=eid,
                gene_id=gene_id,
                binsizes=binsizes,
                use_marks=MARKS,
            )
        except FileNotFoundError:
            print(f"[Warn] npy not found for gene {gene_id} (eid={eid}), skip.")
            continue

        X_list.append(feats)
        y_bf_list.append(bf_label)
        y_bs_list.append(bs_label)
        chrom_list.append(chrom)
        gene_ids.append(gene_id)

    X = np.stack(X_list, axis=0)                # (N, D)
    y_bf = np.array(y_bf_list, dtype=int)       # (N,)
    y_bs = np.array(y_bs_list, dtype=int)       # (N,)
    chroms = np.array(chrom_list, dtype=object) # (N,)

    return X, y_bf, y_bs, chroms, gene_ids