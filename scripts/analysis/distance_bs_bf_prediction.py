import argparse
import torch
import torch.nn as nn
import os
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import seaborn as sns
from src.model.data import DeepBurstDataset
from src.model.net import  DeepBurst
from src.utils.tools import seed_everything
from src.utils.constants import DEVICE
from src.model.constants import get_config

MARKS = ["H3K4me1","H3K4me3","H3K9me3","H3K27me3","H3K36me3","H3K27ac","H3K9ac"]
torch.autograd.set_detect_anomaly(True)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def evaluation(out:'torch.Tensor', label:'torch.Tensor'):
    score = out.softmax(axis=1)[:, 1]
    pred = out.argmax(axis=1)

    acc = metrics.accuracy_score(label, pred) * 100
    auc = metrics.roc_auc_score(label, score) * 100
    ap = metrics.average_precision_score(label, score) * 100
    return score, label, pred, acc, auc, ap


config_path = "configs/default.yaml"
config = get_config(config_path)

predictions = []

for eid in ["E116", "E118", "E003"]:
    config["remove_marks"] = []
    # 注意：data.py 期望 masked_marks 是 dict；你这里不做 mask 就给空 dict
    config["masked_marks"] = {}

    seed = config["seed"]
    bsz  = config["bsz"]

    i_max = config["i_max"]
    w_max = config["w_max"]          # 这个决定 max_n_bins（padding 后固定长度）
    n_feats_p = config['promoter_feats_basic_nums'] - len(config["remove_marks"])

    d_emb = config["embed"]["d_model"]
    embed_kws = config["embed"]
    d_head = config["d_head"]

    targets = ['bs_label', 'bf_label']
    npy_dir = "extra/datasets/processed/v1"

    binsizes = [500]

    # 关键：直接用 w_prom 控制 window（data.py 会以 20kb 为中心裁剪）
    for window_size in [1000, 2000, 5000, 10000, 20000, 40000]:
        # 让 config 里也同步（如果你其它地方会读 config["w_prom"]）
        config["w_prom"] = window_size
        w_prom = window_size

        for fold in [0, 1, 2, 3]:
            print(f"eid:{eid}, fold:{fold}, window:{window_size}")

            checkpoints = f"checkpoints/{eid}.{fold}.bs_bf_para.model.pt"
            meta_path = f"extra/datasets/processed/v2/meta_datasets/meta_data_{eid}.csv"

            seed_everything(seed)
            meta = pd.read_csv(meta_path).sample(frac=1, random_state=seed).reset_index(drop=True)

            splits = {
                1: ['chr1', 'chr6', 'chr5', 'chr8', 'chr14', 'chrY'],
                2: ['chr7', 'chr10', 'chr11', 'chr12', 'chr15', 'chr21'],
                3: ['chr2', 'chr3', 'chr4', 'chr16', 'chr18', 'chr20'],
                4: ['chr9', 'chr13', 'chr17', 'chr19', 'chr22', 'chrX'],
            }
            chromosome_splits = {chrom: k for k, chroms in splits.items() for chrom in chroms}

            qs = [
                meta[meta.apply(lambda row: chromosome_splits[row['chrom']] == 1, axis=1)].gene_id.tolist(),
                meta[meta.apply(lambda row: chromosome_splits[row['chrom']] == 2, axis=1)].gene_id.tolist(),
                meta[meta.apply(lambda row: chromosome_splits[row['chrom']] == 3, axis=1)].gene_id.tolist(),
                meta[meta.apply(lambda row: chromosome_splits[row['chrom']] == 4, axis=1)].gene_id.tolist(),
            ]

            train_genes = qs[(fold + 0) % 4] + qs[(fold + 1) % 4] + qs[(fold + 2) % 4]
            val_genes   = qs[(fold + 3) % 4]

            val_dataset = DeepBurstDataset(
                meta_path,
                npy_dir,
                val_genes,
                i_max,
                binsizes,
                w_prom,      # 关键：这里传入 window_size
                w_max,       # 建议保持 40000，让输入维度固定为 80 bins（binsize=500）
                targets=targets,
                config=config,
                with_gene_id=True
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
                for batch, d in bar:
                    gene_ids += d.pop('gene_id')
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

            val_preds = {t: p for t, p in zip(targets, torch.chunk(val_out, len(targets), axis=-1))}
            val_labels = {t: y for t, y in zip(targets, torch.chunk(val_label, len(targets), axis=-1))}

            records = {"gene_id": gene_ids}
            description = ""

            for target in targets:
                val_score, val_lab, val_pred, val_acc, val_auc, val_ap = evaluation(val_preds[target], val_labels[target])
                records[f'{target}_label'] = val_lab.cpu().numpy().squeeze()
                records[f'{target}_score'] = val_score.cpu().numpy().squeeze()
                records[f'{target}_pred']  = val_pred.cpu().numpy().squeeze()
                description += f"{target}: acc={val_acc:.4f}, auc={val_auc:.4f}, ap={val_ap:.4f} "

            records["eid"] = eid
            records["fold"] = fold
            records["window_size"] = window_size

            predictions.append(pd.DataFrame(records))
            print(description)

print("All done.")
predictions = pd.concat(predictions, axis=0)
predictions.to_csv("extra/results/distance_region_predictions_bs_bf.csv", index=False)