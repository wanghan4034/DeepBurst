import argparse
import torch
import torch.nn as nn
import os
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import seaborn as sns

from src.model.data import DeepBurstDataset
from src.model.net import DeepBurst
from src.utils.tools import seed_everything
from src.utils.constants import DEVICE
from src.model.constants import get_config, PERTURBATION_STRENGTH, PERTURBATION_REGION

MARKS = ["H3K4me1","H3K4me3","H3K9me3","H3K27me3","H3K36me3","H3K27ac","H3K9ac"]
torch.autograd.set_detect_anomaly(True)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def evaluation(out: torch.Tensor, label: torch.Tensor):
    score = out.softmax(axis=1)[:, 1]
    pred = out.argmax(axis=1)

    acc = metrics.accuracy_score(label, pred) * 100
    auc = metrics.roc_auc_score(label, score) * 100
    ap  = metrics.average_precision_score(label, score) * 100
    return score, label, pred, acc, auc, ap

config_path = "configs/default.yaml"
config = get_config(config_path)

for eid in ["E116"]:
    predictions = []

    for keep_mark in MARKS:
        print(f"EID:{eid}, keep_mark:{keep_mark}")

        keep_marks = [keep_mark]
        remove_marks = [m for m in MARKS if m not in keep_marks]

        if remove_marks:
            config["remove_marks"] = remove_marks
            model_tag = "_".join(remove_marks)
        else:
            config["remove_marks"] = []
            model_tag = None

        seed = config["seed"]
        bsz = config["bsz"]

        i_max = config["i_max"]
        w_prom = config["w_prom"]
        w_max = config["w_max"]

        # 注意：这里假设 promoter_feats_basic_nums 对应“marks 数”
        n_feats_p = config["promoter_feats_basic_nums"] - len(config["remove_marks"])
        d_emb = config["embed"]["d_model"]
        embed_kws = config["embed"]
        d_head = config["d_head"]

        targets = ["mean_label"]
        npy_dir = "extra/datasets/processed/v1"
        binsizes = [500]

        # data.py 使用 PERTURBATION_REGION 控制 bins；marked_bin_idxes 这个字段不再需要
        marked_bin_idxes = list(range(80))  # 或者使用 "all"

        for perturbation_strength in range(11):
            strength = perturbation_strength * 0.1
            config["perturbation_strength"] = strength  # 保留：即便 data.py 不用也无妨

            # -----------------------------
            # 关键：适配 data.py 的 masked_marks 结构（dict）
            # -----------------------------
            config["masked_marks"] = {
                keep_mark: {
                    PERTURBATION_STRENGTH: strength,
                    PERTURBATION_REGION: marked_bin_idxes,  # 或 "all"
                }
            }

            for fold in [0, 1, 2, 3]:
                print(f"eid:{eid}, fold:{fold}")

                if remove_marks:
                    checkpoints = f"checkpoints/{eid}.remove_{model_tag}.{fold}.mean_para.model.pt"
                else:
                    checkpoints = f"checkpoints/{eid}.{fold}.mean_para.model.pt"

                meta_path = f"extra/datasets/processed/v2/meta_datasets/meta_data_{eid}.csv"

                seed_everything(seed)
                meta = pd.read_csv(meta_path).sample(frac=1, random_state=seed).reset_index(drop=True)

                splits = {
                    1: ["chr1", "chr6", "chr5", "chr8", "chr14", "chrY"],
                    2: ["chr7", "chr10", "chr11", "chr12", "chr15", "chr21"],
                    3: ["chr2", "chr3", "chr4", "chr16", "chr18", "chr20"],
                    4: ["chr9", "chr13", "chr17", "chr19", "chr22", "chrX"],
                }
                chromosome_splits = {chrom: k for k, chroms in splits.items() for chrom in chroms}

                qs = [
                    meta[meta.apply(lambda row: chromosome_splits[row["chrom"]] == 1, axis=1)].gene_id.tolist(),
                    meta[meta.apply(lambda row: chromosome_splits[row["chrom"]] == 2, axis=1)].gene_id.tolist(),
                    meta[meta.apply(lambda row: chromosome_splits[row["chrom"]] == 3, axis=1)].gene_id.tolist(),
                    meta[meta.apply(lambda row: chromosome_splits[row["chrom"]] == 4, axis=1)].gene_id.tolist(),
                ]

                train_genes = qs[(fold + 0) % 4] + qs[(fold + 1) % 4] + qs[(fold + 2) % 4]
                val_genes   = qs[(fold + 3) % 4]

                val_dataset = DeepBurstDataset(
                    meta_path,
                    npy_dir,
                    val_genes,
                    i_max,
                    binsizes,
                    w_prom,
                    w_max,
                    targets=targets,
                    config=config,
                    with_gene_id=True,
                )
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz)

                model = DeepBurst(
                    n_feats_p,
                    d_emb,
                    d_head,
                    embed_kws=embed_kws,
                    binsizes=binsizes,
                    seed=42,
                    targets=targets,
                ).to(DEVICE)

                ckpt = torch.load(checkpoints, map_location=DEVICE)
                model.load_state_dict(ckpt["net"])

                bar = tqdm(enumerate(val_loader, 1), total=len(val_loader))
                gene_ids, val_out, val_label = [], [], []

                model.eval()
                with torch.no_grad():
                    for _, d in bar:
                        gene_ids += d.pop("gene_id")

                        # 将 batch 中的 tensor / dict[tensor] 放到 DEVICE
                        for k, v in d.items():
                            if isinstance(v, dict):
                                for _k, _v in v.items():
                                    v[_k] = _v.to(DEVICE)
                            else:
                                d[k] = v.to(DEVICE)

                        out = model(d["promoter_feats"][500])
                        val_out.append(out.cpu())
                        val_label.append(d["label"].cpu())

                val_out = torch.cat(val_out)
                val_label = torch.cat(val_label)

                val_preds = {}
                for target, pred in zip(targets, torch.chunk(val_out, len(targets), axis=-1)):
                    val_preds[target] = pred

                val_labels = {}
                for target, label in zip(targets, torch.chunk(val_label, len(targets), axis=-1)):
                    val_labels[target] = label

                description = ""
                records = {"gene_id": gene_ids}

                for target in targets:
                    val_score, y_true, y_pred, val_acc, val_auc, val_ap = evaluation(
                        val_preds[target], val_labels[target]
                    )

                    records[f"{target}_label"] = y_true.cpu().numpy().squeeze()
                    records[f"{target}_score"] = val_score.cpu().numpy().squeeze()
                    records[f"{target}_pred"]  = y_pred.cpu().numpy().squeeze()
                    description += f"{target}: acc={val_acc:.4f}, auc={val_auc:.4f}, ap={val_ap:.4f} "

                records["fold"] = fold
                records["keep_mark"] = keep_mark
                records["perturbation_strength"] = strength  # 用真实 float 强度

                predictions.append(pd.DataFrame(records))
                print(description)

    print(f"EID:{eid} done.")
    predictions = pd.concat(predictions, axis=0)
    predictions.to_csv(f"extra/results/{eid}_perturbation_predictions_mean.csv", index=False)

if __name__ == "__main__":
    print("Done")