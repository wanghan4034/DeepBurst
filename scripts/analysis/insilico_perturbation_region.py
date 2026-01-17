#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy
import torch
import pandas as pd
from tqdm import tqdm
from sklearn import metrics

from src.model.data import DeepBurstDataset
from src.model.net import DeepBurst
from src.utils.tools import seed_everything
from src.utils.constants import DEVICE
from src.model.constants import (
    get_config, MARKS, PERTURBATION_STRENGTH, PERTURBATION_REGION
)

# ===================== 用户参数 ===================== #
# 细胞类型与折数
EIDS              = ["E116", "E118", "E003"]
FOLDS             = [0, 1, 2, 3]

# 数据与模型路径
META_DIR          = "extra/datasets/processed/v1/meta_datasets"
REGION_DIR        = "extra/datasets/processed/v1"                    # npy 根目录
CHECKPOINT_DIR    = "checkpoints"

# 配置文件
CONFIG_PATH       = "configs/default.yaml"

# 训练/推理常用
BATCH_SIZE        = None   # 若为 None，使用 config 里的 bsz
SEED              = None   # 若为 None，使用 config 里的 seed

# 扰动与滑动窗口（单位：bin；500bp/bin）
BINSIZE_BP        = 500
SLIDE_K_BINS      = 1      # 窗口宽度；=1 表示单 bin（500bp）
STRIDE_BINS       = 1      # 步长；=1 表示每 500bp 移动一次
MASK_STRENGTH     = 0.0    # 0=完全抹除；0.5=减半；>1=增强

# 输出
OUT_CSV           = f"extra/results/slide_perturbation_{BINSIZE_BP}bp_k{SLIDE_K_BINS}_s{STRIDE_BINS}_ALL_MARKS.csv"

# 是否保存预测的 argmax（可选）
SAVE_PRED         = True

# ===================== 工具函数 ===================== #
def softmax_score(out: torch.Tensor) -> torch.Tensor:
    """二分类：返回类别1的概率(score)"""
    return out.softmax(dim=1)[:, 1]

# ---------------- eval helper ---------------- #
def evaluation(out: torch.Tensor, label: torch.Tensor):
    score = out.softmax(dim=1)[:, 1]
    pred  = out.argmax(dim=1)
    acc   = metrics.accuracy_score(label.numpy(), pred.numpy()) * 100
    auc   = metrics.roc_auc_score(label.numpy(), score.numpy()) * 100
    ap    = metrics.average_precision_score(label.numpy(), score.numpy()) * 100
    return score, label, pred, acc, auc, ap

# ===================== 主流程 ===================== #
def main():
    # 读取基础配置
    config = get_config(CONFIG_PATH)
    seed = SEED if SEED is not None else config["seed"]
    bsz  = BATCH_SIZE if BATCH_SIZE is not None else config["bsz"]

    # 固定 binsize = 500
    binsizes = [BINSIZE_BP]

    # 常规配置清理
    config["remove_marks"]   = [] if not config.get("remove_marks") else config["remove_marks"]
    config["marked_bin_idxes"] = []
    config["masked_marks"]   = {}  # 运行时会覆盖

    # 一些模型尺寸参数
    
    
    n_feats_p = (
        config['promoter_feats_basic_nums'] - len(config["remove_marks"]) + feature_bin_kws['out_channels']
        if add_feature_bin else
        config['promoter_feats_basic_nums'] - len(config["remove_marks"])
    )
    d_emb   = config["embed"]["d_model"]
    d_head  = config["d_head"]
    targets = ['bs_label', 'bf_label']
    i_max   = config["i_max"]
    w_prom  = config["w_prom"]       # 例如 40000
    w_max   = config["w_max"]

    seed_everything(seed)

    all_records = []

    for eid in EIDS:
        for fold in FOLDS:
            print(f"\n==== eid:{eid}, fold:{fold} ====")

            # 路径
            meta_path  = os.path.join(META_DIR, f"meta_data_{eid}.csv")
            ckpt_path  = os.path.join(CHECKPOINT_DIR, f"{eid}.{fold}.bs_bf_para.model.pt")

            # 读 meta 并构建染色体分组
            meta = pd.read_csv(meta_path).sample(frac=1, random_state=seed).reset_index(drop=True)
            splits = {
                1: ['chr1','chr6','chr5','chr8','chr14','chrY'],
                2: ['chr7','chr10','chr11','chr12','chr15','chr21'],
                3: ['chr2','chr3','chr4','chr16','chr18','chr20'],
                4: ['chr9','chr13','chr17','chr19','chr22','chrX'],
            }
            chrom2split = {c: k for k, arr in splits.items() for c in arr}
            qs = [
                meta[meta.apply(lambda row: chrom2split[row['chrom']] == 1, axis=1)].gene_id.tolist(),
                meta[meta.apply(lambda row: chrom2split[row['chrom']] == 2, axis=1)].gene_id.tolist(),
                meta[meta.apply(lambda row: chrom2split[row['chrom']] == 3, axis=1)].gene_id.tolist(),
                meta[meta.apply(lambda row: chrom2split[row['chrom']] == 4, axis=1)].gene_id.tolist(),
            ]
            val_genes = qs[(fold + 3) % 4]
            print(f"Val gene count: {len(val_genes)} | n_feats_p={n_feats_p}")

            # 模型
            model = DeepBurst(
                n_feats_p, d_emb, d_head,
                embed_kws=config["embed"], binsizes=binsizes, seed=42, targets=targets
            ).to(DEVICE)
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(ckpt["net"])
            model.eval()

            # baseline：不遮挡
            run_cfg = copy.deepcopy(config)
            run_cfg["masked_marks"] = {}
            val_dataset = DeepBurstDataset(
                meta_path, REGION_DIR, val_genes,
                i_max, binsizes, w_prom, w_max,
                targets=targets, config=run_cfg, with_gene_id=True
            )
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz)
            bar = tqdm(enumerate(val_loader, 1), total=len(val_loader), leave=False, desc="No_Mask")
            gene_ids, outs, labels = [], [], []
            with torch.no_grad():
                for _, batch in bar:
                    gene_ids += batch.pop('gene_id')
                    # to device
                    for k, v in batch.items():
                        if isinstance(v, dict):
                            for _k, _v in v.items():
                                v[_k] = _v.to(DEVICE)
                        else:
                            batch[k] = v.to(DEVICE)
                    out = model(
                        batch["promoter_feats"][binsizes[0]],
                        batch["promoter_pad_masks"][binsizes[0]],
                    )
                    outs.append(out.cpu())
                    labels.append(batch["label"].cpu())

            out_t   = torch.cat(outs)        # (N, 2*targets)
            label_t = torch.cat(labels)
            rec = {"gene_id": gene_ids}
            chunks_out   = torch.chunk(out_t, len(targets), dim=-1)
            chunks_label = torch.chunk(label_t, len(targets), dim=-1)
            desc_line = ["No_Mask"]
            for tname, logits, lab in zip(targets, chunks_out, chunks_label):
                s, y, p, acc, auc, ap = evaluation(logits, lab)
                rec[f"{tname}_score"] = s.numpy().squeeze()
                rec[f"{tname}_label"] = y.numpy().squeeze()
                rec[f"{tname}_pred"]  = p.numpy().squeeze()
                rec[f"{tname}_acc"]   = acc
                rec[f"{tname}_auc"]   = auc
                rec[f"{tname}_ap"]    = ap
                desc_line.append(f"{tname}: acc={acc:.2f}, auc={auc:.2f}, ap={ap:.2f}")
                if SAVE_PRED:
                    pred = logits.argmax(dim=1).numpy()
                    rec[f"{tname}_pred"] = pred

            print(f"[{eid}][fold={fold}] No_Mask | " + " | ".join(desc_line))
            # 元信息
            rec["eid"]           = [eid] * len(gene_ids)
            rec["fold"]          = [fold] * len(gene_ids)
            rec["masked_marks"]  = ["None"] * len(gene_ids)
            rec["mask_strength"] = [1] * len(gene_ids)
            rec["binsize"]       = [binsizes[0]] * len(gene_ids)
            rec["slide_k"]       = [0] * len(gene_ids)
            rec["stride"]        = [0] * len(gene_ids)
            rec["bin_start"]     = [0] * len(gene_ids)
            rec["bin_end"]       = [0] * len(gene_ids)
            rec["rel_start_bp"]  = [0] * len(gene_ids)
            rec["rel_end_bp"]    = [0] * len(gene_ids)
            all_records.append(pd.DataFrame(rec))
            # 以上为 baseline

            # 距离滑动
            binsize   = binsizes[0]           # 500
            max_bins  = w_prom // binsize     # 例如 40000//500 = 80
            half_bp   = w_prom // 2
            starts    = list(range(0, max_bins - SLIDE_K_BINS + 1, STRIDE_BINS))

            for bin_start in starts:
                region_bins   = list(range(bin_start, bin_start + SLIDE_K_BINS))
                rel_start_bp  = (bin_start * binsize) - half_bp
                rel_end_bp    = ((bin_start + SLIDE_K_BINS) * binsize) - half_bp

                # 统一掩码：所有 marks 同一窗口
                run_cfg = copy.deepcopy(config)
                run_cfg["masked_marks"] = {
                    m: {PERTURBATION_REGION: region_bins,
                        PERTURBATION_STRENGTH: float(MASK_STRENGTH)}
                    for m in MARKS
                }

                # DataLoader
                val_dataset = DeepBurstDataset(
                    meta_path, REGION_DIR, val_genes,
                    i_max, [binsize], w_prom, w_max,
                    targets=targets, config=run_cfg, with_gene_id=True
                )
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz)

                # 推理
                bar = tqdm(enumerate(val_loader, 1), total=len(val_loader), leave=False,
                           desc=f"ALL_MARKS bins[{bin_start}:{bin_start+SLIDE_K_BINS})")
                gene_ids, outs, labels = [], [], []
                with torch.no_grad():
                    for _, batch in bar:
                        gene_ids += batch.pop('gene_id')
                        # to device
                        for k, v in batch.items():
                            if isinstance(v, dict):
                                for _k, _v in v.items():
                                    v[_k] = _v.to(DEVICE)
                            else:
                                batch[k] = v.to(DEVICE)
                        out = model(
                            batch["promoter_feats"][binsize],
                            batch["promoter_pad_masks"][binsize],
                        )
                        outs.append(out.cpu())
                        labels.append(batch["label"].cpu())

                # 拼接
                out_t   = torch.cat(outs)        # (N, 2*targets)
                label_t = torch.cat(labels)      # (N, 2)

                # 拆每个 target 的 score/label（无需计算AUC/ACC）
                rec = {"gene_id": gene_ids}
                chunks_out   = torch.chunk(out_t, len(targets), dim=-1)
                chunks_label = torch.chunk(label_t, len(targets), dim=-1)
                desc_line = [f"ALL_MARKS bins[{bin_start}:{bin_start+SLIDE_K_BINS})"]
                for tname, logits, lab in zip(targets, chunks_out, chunks_label):
                    s, y, p, acc, auc, ap = evaluation(logits, lab)
                    rec[f"{tname}_score"] = s.numpy().squeeze()
                    rec[f"{tname}_label"] = y.numpy().squeeze()
                    rec[f"{tname}_pred"]  = p.numpy().squeeze()
                    rec[f"{tname}_acc"]   = acc
                    rec[f"{tname}_auc"]   = auc
                    rec[f"{tname}_ap"]    = ap
                    desc_line.append(f"{tname}: acc={acc:.2f}, auc={auc:.2f}, ap={ap:.2f}")
                    if SAVE_PRED:
                            pred = logits.argmax(dim=1).numpy()
                            rec[f"{tname}_pred"] = pred

                print(f"[{eid}][fold={fold}] bin={bin_start} | " + " | ".join(desc_line))
                   

                # 元信息
                rec["eid"]           = [eid] * len(gene_ids)
                rec["fold"]          = [fold] * len(gene_ids)
                rec["masked_marks"]  = ["ALL"] * len(gene_ids)
                rec["mask_strength"] = [MASK_STRENGTH] * len(gene_ids)
                rec["binsize"]       = [binsize] * len(gene_ids)
                rec["slide_k"]       = [SLIDE_K_BINS] * len(gene_ids)
                rec["stride"]        = [STRIDE_BINS] * len(gene_ids)
                rec["bin_start"]     = [bin_start] * len(gene_ids)
                rec["bin_end"]       = [bin_start + SLIDE_K_BINS] * len(gene_ids)
                rec["rel_start_bp"]  = [rel_start_bp] * len(gene_ids)
                rec["rel_end_bp"]    = [rel_end_bp] * len(gene_ids)

                all_records.append(pd.DataFrame(rec))

    # 汇总保存
    df_all = pd.concat(all_records, axis=0).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df_all.to_csv(OUT_CSV, index=False)
    print("Saved to:", OUT_CSV)


if __name__ == "__main__":
    main()