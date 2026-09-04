import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn import metrics

from src.model.net import DeepBurst
from src.utils.tools import seed_everything
from src.utils.constants import DEVICE
from src.model.constants import get_config

from benchmarks.revised_v2.reviewer1.data_extractor import HistoneExtractor


torch.autograd.set_detect_anomaly(True)

# infer_methods = '_deeptx'
infer_methods = ''
# ============================================================
# 1. Basic config
# ============================================================

config_path = "configs/default.yaml"
config = get_config(config_path)

remove_marks = None
# remove_marks = "H3K36me3"

if remove_marks:
    config["remove_marks"] = [remove_marks] if isinstance(remove_marks, str) else list(remove_marks)
else:
    config["remove_marks"] = []

config["marked_bin_idxes"] = []
config["masked_marks"] = []

seed = config["seed"]
seed_everything(seed)

bsz = config["bsz"]

n_feats_p = config["promoter_feats_basic_nums"] - len(config["remove_marks"])
d_emb = config["embed"]["d_model"]
d_head = config["d_head"]
embed_kws = config["embed"]

targets = ["bs_label", "bf_label"]
binsizes = [500]

ensg2tss_path = "extra/datasets/annotations/ensg2tss.pickle"
bigwig_dir = "extra/datasets/epigenetic/hg19"
meta_dir = "extra/datasets/processed/v2/meta_datasets"

result_dir = Path("benchmarks/revised_v2/reviewer1/results")
result_dir.mkdir(parents=True, exist_ok=True)

eids = ["E003", "E116", "E118"]
batch_size = 32
strand_oriented = True


# ============================================================
# 2. Chromosome split logic
# ============================================================

def build_chromosome_splits():
    splits = {
        1: ["chr1", "chr6", "chr5", "chr8", "chr14", "chrY"],
        2: ["chr7", "chr10", "chr11", "chr12", "chr15", "chr21"],
        3: ["chr2", "chr3", "chr4", "chr16", "chr18", "chr20"],
        4: ["chr9", "chr13", "chr17", "chr19", "chr22", "chrX"],
    }

    chromosome_splits = {}
    for split_id, chroms in splits.items():
        for chrom in chroms:
            chromosome_splits[chrom] = split_id

    return chromosome_splits


CHROMOSOME_SPLITS = build_chromosome_splits()


def chrom_to_fold(chrom):
    """
    Match training code:

        val_genes = qs[(fold + 3) % 4]

    Therefore:
        fold 0 -> split 4
        fold 1 -> split 1
        fold 2 -> split 2
        fold 3 -> split 3

    So:
        split_id -> fold = split_id % 4
    """
    if chrom not in CHROMOSOME_SPLITS:
        raise ValueError(f"Unknown chromosome: {chrom}")

    split_id = CHROMOSOME_SPLITS[chrom]
    return split_id % 4


# ============================================================
# 3. Model loading
# ============================================================

def get_checkpoint_path(eid, fold, config, remove_marks=None):
    if remove_marks:
        remove_name = "_".join(config["remove_marks"])
        ckpt_path = f"checkpoints/{eid}.remove_{remove_name}.{fold}.bs_bf_para.model.pt"
    else:
        ckpt_path = f"checkpoints/{eid}.{fold}.bs_bf_para.model.pt"

    return ckpt_path


def build_deepburst_model():
    model = DeepBurst(
        n_feats_p,
        d_emb,
        d_head,
        embed_kws=embed_kws,
        binsizes=binsizes,
        seed=42,
        targets=targets,
    ).to(DEVICE)

    return model


def load_deepburst_model(eid, fold, config, remove_marks=None):
    model = build_deepburst_model()

    ckpt_path = get_checkpoint_path(
        eid=eid,
        fold=fold,
        config=config,
        remove_marks=remove_marks,
    )

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["net"])
    model.eval()

    return model, ckpt_path


def load_all_fold_models(eid, config, remove_marks=None, folds=(0, 1, 2, 3)):
    model_dict = {}

    for fold in folds:
        print(f"[Load model] eid={eid}, fold={fold}")

        model, ckpt_path = load_deepburst_model(
            eid=eid,
            fold=fold,
            config=config,
            remove_marks=remove_marks,
        )

        model_dict[int(fold)] = {
            "model": model,
            "checkpoint": ckpt_path,
        }

    return model_dict


# ============================================================
# 4. Tensor helpers
# ============================================================

def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)

    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}

    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]

    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)

    return obj


def subset_model_inputs(model_inputs, idx):
    if torch.is_tensor(model_inputs):
        return model_inputs[idx]

    if isinstance(model_inputs, dict):
        return {k: subset_model_inputs(v, idx) for k, v in model_inputs.items()}

    if isinstance(model_inputs, list):
        return [subset_model_inputs(v, idx) for v in model_inputs]

    if isinstance(model_inputs, tuple):
        return tuple(subset_model_inputs(v, idx) for v in model_inputs)

    return model_inputs


def make_model_inputs_from_X(X, bin_size=500):
    """
    Convert extracted histone matrix to DeepBurst input.

    X:
        np.ndarray, shape = (B, 80, 7)

    Return:
        model_inputs["promoter_feats"][500]:
            torch.Tensor, shape = (B, 1, 80, 7)
    """
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)

    X_tensor = torch.from_numpy(X).float()

    if X_tensor.ndim != 3:
        raise ValueError(
            f"Expected X shape = (B, n_bins, n_marks), got {X_tensor.shape}"
        )

    model_inputs = {
        "promoter_feats": {
            bin_size: X_tensor.unsqueeze(1)
        }
    }

    return model_inputs


# ============================================================
# 5. Parse DeepBurst output
# ============================================================

def parse_deepburst_output(out, targets=("bs_label", "bf_label")):
    """
    Follow validation logic:

        torch.chunk(val_out, len(targets), axis=-1)

    Each target has two logits.
    """
    pred_chunks = torch.chunk(out, len(targets), dim=-1)

    records = {}

    for target, logits in zip(targets, pred_chunks):
        score = logits.softmax(dim=1)[:, 1]
        pred = logits.argmax(dim=1)

        records[f"{target}_score"] = score.detach().cpu().numpy()
        records[f"{target}_pred"] = pred.detach().cpu().numpy()

    return pd.DataFrame(records)


# ============================================================
# 6. Prediction helpers
# ============================================================

def batched_gene_ids(gene_ids, batch_size=8):
    for i in range(0, len(gene_ids), batch_size):
        yield gene_ids[i:i + batch_size]


def predict_gene_batch_by_fold(
    gene_ids,
    eid,
    extractor,
    model_dict,
    strand_oriented=True,
):
    """
    Predict one batch of genes for one eid.
    """
    if isinstance(gene_ids, str):
        gene_ids = [gene_ids]
    else:
        gene_ids = list(gene_ids)

    X, meta_df = extractor.get_gene_matrices(
        gene_ids,
        strand_oriented=strand_oriented,
    )

    if meta_df.empty:
        return pd.DataFrame()

    model_inputs = make_model_inputs_from_X(X, bin_size=500)

    meta_df = meta_df.reset_index(drop=True).copy()
    meta_df["fold"] = meta_df["chrom"].apply(chrom_to_fold)

    all_results = []

    for fold, sub_meta in meta_df.groupby("fold"):
        fold = int(fold)

        model = model_dict[fold]["model"]
        ckpt_path = model_dict[fold]["checkpoint"]

        idx = torch.tensor(sub_meta.index.to_numpy(), dtype=torch.long)

        fold_model_inputs = subset_model_inputs(model_inputs, idx)
        fold_model_inputs = move_to_device(fold_model_inputs, DEVICE)

        fold_x = fold_model_inputs["promoter_feats"][500]

        model.eval()
        with torch.no_grad():
            out = model(fold_x)

        pred_part = parse_deepburst_output(out, targets=targets)

        result_part = sub_meta.reset_index(drop=True).copy()
        result_part = pd.concat([result_part, pred_part], axis=1)

        result_part["eid"] = eid
        result_part["checkpoint"] = ckpt_path

        all_results.append(result_part)

    if len(all_results) == 0:
        return pd.DataFrame()

    pred_df = pd.concat(all_results, axis=0).reset_index(drop=True)

    return pred_df


# ============================================================
# 7. AUC helpers
# ============================================================

def attach_labels(pred_df, meta):
    label_cols = ["gene_id", "bs_label", "bf_label"]

    meta_label = (
        meta[label_cols]
        .drop_duplicates("gene_id")
        .copy()
    )

    pred_labeled_df = pred_df.merge(
        meta_label,
        on="gene_id",
        how="left",
        validate="many_to_one",
    )

    return pred_labeled_df


def compute_auc_table(pred_labeled_df, eid):
    """
    Return AUC table with columns:
        eid, fold, auc, target

    Example:
        E116,0,91.61,bs_label
        E116,0,89.96,bf_label
    """
    rows = []

    for fold, sub in pred_labeled_df.groupby("fold"):
        fold = int(fold)

        for target in ["bs_label", "bf_label"]:
            score_col = f"{target}_score"

            valid = sub[[target, score_col]].dropna().copy()

            if valid.empty or valid[target].nunique() < 2:
                auc = np.nan
            else:
                auc = metrics.roc_auc_score(
                    valid[target].astype(int),
                    valid[score_col].astype(float),
                ) * 100

            rows.append({
                "eid": eid,
                "fold": fold,
                "auc": auc,
                "target": target,
            })

    auc_df = pd.DataFrame(rows)

    auc_df = auc_df.sort_values(
        ["eid", "fold", "target"],
        ascending=[True, True, False],  # 让 bs_label 通常排在 bf_label 前面可能不稳定
    ).reset_index(drop=True)

    # 强制按照 bs_label, bf_label 顺序
    target_order = pd.Categorical(
        auc_df["target"],
        categories=["bs_label", "bf_label"],
        ordered=True,
    )
    auc_df["target"] = target_order

    auc_df = auc_df.sort_values(["eid", "fold", "target"]).reset_index(drop=True)
    auc_df["target"] = auc_df["target"].astype(str)

    return auc_df[["eid", "fold", "auc", "target"]]


# ============================================================
# 8. Run all EIDs
# ============================================================

all_pred_labeled = []
all_auc = []

for eid in eids:
    print("=" * 80)
    print(f"[START] eid={eid}")
    print("=" * 80)

    meta_path = f"{meta_dir}/meta_data_{eid}{infer_methods}.csv"
    meta = pd.read_csv(meta_path)

    gene_ids_all = (
        meta["gene_id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    print(f"[INFO] eid={eid}, total genes={len(gene_ids_all)}")
    print(f"[INFO] batch size={batch_size}")
    print(f"[INFO] number of batches={int(np.ceil(len(gene_ids_all) / batch_size))}")

    # Load all fold models once for this eid
    model_dict = load_all_fold_models(
        eid=eid,
        config=config,
        remove_marks=remove_marks,
        folds=(0, 1, 2, 3),
    )

    # Create extractor once for this eid
    extractor = HistoneExtractor(
        ensg2tss_path=ensg2tss_path,
        bigwig_dir=bigwig_dir,
        eid=eid,
        seed=seed,
        log1p=True,
    )

    pred_batches = []

    for batch_gene_ids in tqdm(
        list(batched_gene_ids(gene_ids_all, batch_size=batch_size)),
        desc=f"Predict {eid}",
    ):
        batch_pred_df = predict_gene_batch_by_fold(
            gene_ids=batch_gene_ids,
            eid=eid,
            extractor=extractor,
            model_dict=model_dict,
            strand_oriented=strand_oriented,
        )

        if not batch_pred_df.empty:
            pred_batches.append(batch_pred_df)

    extractor.close()

    if len(pred_batches) == 0:
        print(f"[WARN] No predictions generated for {eid}")
        continue

    pred_df = pd.concat(pred_batches, axis=0).reset_index(drop=True)

    pred_labeled_df = attach_labels(pred_df, meta)

    print(f"[SUMMARY] eid={eid}")
    print("Predictions:", pred_df.shape[0])
    print("After label merge:", pred_labeled_df.shape[0])
    print("Missing bs_label:", pred_labeled_df["bs_label"].isna().sum())
    print("Missing bf_label:", pred_labeled_df["bf_label"].isna().sum())

    auc_df = compute_auc_table(pred_labeled_df, eid)

    print("[AUC]")
    print(auc_df[auc_df["fold"].astype(str) == "ALL"])

    # Save per-eid results
    pred_out = result_dir / f"random_tss_selected_gene_predictions_by_fold_{eid}{infer_methods}.csv"
    auc_out = result_dir / f"random_tss_auc_{eid}{infer_methods}.csv"

    pred_labeled_df.to_csv(pred_out, index=False)
    auc_df.to_csv(auc_out, index=False)

    print(f"[SAVED] {pred_out}")
    print(f"[SAVED] {auc_out}")

    all_pred_labeled.append(pred_labeled_df)
    all_auc.append(auc_df)

    # Release models for this eid
    del model_dict
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ============================================================
# 9. Save combined results
# ============================================================

if len(all_pred_labeled) > 0:
    all_pred_labeled_df = pd.concat(all_pred_labeled, axis=0).reset_index(drop=True)
    all_pred_out = result_dir / f"random_tss_selected_gene_predictions_by_fold_ALL_EIDS{infer_methods}.csv"
    all_pred_labeled_df.to_csv(all_pred_out, index=False)
    print(f"[SAVED] {all_pred_out}")

if len(all_auc) > 0:
    all_auc_df = pd.concat(all_auc, axis=0).reset_index(drop=True)
    all_auc_out = result_dir / f"random_tss_auc_ALL_EIDS{infer_methods}.csv"
    all_auc_df.to_csv(all_auc_out, index=False)
    print(f"[SAVED] {all_auc_out}")

print("[DONE]")