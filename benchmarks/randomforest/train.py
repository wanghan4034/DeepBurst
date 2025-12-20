import os
import numpy as np
import pandas as pd
from typing import Dict

from benchmarks.randomforest.data import build_feature_matrix
from benchmarks.randomforest.model import train_and_eval_classifier


def main():
    # 路径按你的项目调整
    eid = 'E118'
    tag = 'deeptx_delay_1'
    # meta_path = f"extra/datasets/processed/v1/meta_datasets/meta_data_{eid}.csv"
    meta_path = f"extra/datasets/processed/v1/meta_datasets/meta_data_{eid}_{tag}.csv"    
    npy_dir = "/Volumes/ExtremeSSD/BioStudy/CodeReview/DeepBurst/extra/datasets/processed/v1"
    binsizes = [500]
    print(f"Processing EID: {eid}")

    # 构建全基因特征和标签
    X, y_bf, y_bs, chroms, gene_ids = build_feature_matrix(
        meta_path=meta_path,
        npy_dir=npy_dir,
        binsizes=binsizes,
        bf_label_col="bf_label",
        bs_label_col="bs_label",
    )

    print("Feature matrix shape:", X.shape)
    print("bf_label unique:", np.unique(y_bf))
    print("bs_label unique:", np.unique(y_bs))

    # 染色体分组
    splits = {
        1: ['chr1', 'chr6', 'chr5', 'chr8', 'chr14', 'chrY'],
        2: ['chr7', 'chr10', 'chr11', 'chr12', 'chr15', 'chr21'],
        3: ['chr2', 'chr3', 'chr4', 'chr16', 'chr18', 'chr20'],
        4: ['chr9', 'chr13', 'chr17', 'chr19', 'chr22', 'chrX'],
    }

    chromosome_splits: Dict[str, int] = {}
    for key, chrom_list in splits.items():
        for chrom in chrom_list:
            chromosome_splits[chrom] = key

    sample_split_ids = np.array(
        [chromosome_splits.get(c, -1) for c in chroms], dtype=int
    )

    # 收集所有结果
    rows = []

    for split_id in [1, 2, 3, 4]:
        test_mask = sample_split_ids == split_id
        train_mask = ~test_mask

        if test_mask.sum() == 0:
            print(f"[Warn] no samples in test set for split {split_id}, skip.")
            continue

        X_train, X_test = X[train_mask], X[test_mask]
        y_bf_train, y_bf_test = y_bf[train_mask], y_bf[test_mask]
        y_bs_train, y_bs_test = y_bs[train_mask], y_bs[test_mask]

        # === BF 分类 ===
        res_bf = train_and_eval_classifier(
            X_train=X_train,
            y_train=y_bf_train,
            X_test=X_test,
            y_test=y_bf_test,
            task_name="BF label",
            split_id=split_id,
            random_state=42,
        )
        # res_bf 结构: {"rf": {"acc": ..., "auc": ...}, "logit": {...}}
        for method in ["rf", "logit"]:
            metrics = res_bf.get(method, {})
            rows.append({
                "eid": eid,
                "fold": split_id,
                "target": "bf_label",
                "method": method,
                "auc": metrics.get("auc", np.nan),
                "acc": metrics.get("acc", np.nan),
            })

        # === BS 分类 ===
        res_bs = train_and_eval_classifier(
            X_train=X_train,
            y_train=y_bs_train,
            X_test=X_test,
            y_test=y_bs_test,
            task_name="BS label",
            split_id=split_id,
            random_state=42,
        )
        for method in ["rf", "logit"]:
            metrics = res_bs.get(method, {})
            rows.append({
                "eid": eid,
                "fold": split_id,
                "target": "bs_label",
                "method": method,
                "auc": metrics.get("auc", np.nan),
                "acc": metrics.get("acc", np.nan),
            })

    # 写出结果表
    results_df = pd.DataFrame(
        rows,
        columns=["eid", "fold", "target", "method", "auc", "acc"]
    )

    out_path = f"benchmarks/randomforest/results/{eid}_bs_bf_para_{tag}_indicators.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved indicators to: {out_path}")
    print(results_df)


if __name__ == "__main__":
    main()