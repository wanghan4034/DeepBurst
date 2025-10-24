import os
import copy
import itertools
import json
import torch
import pandas as pd
from tqdm import tqdm
from sklearn import metrics

from src.model.data import BurstformerDataset
from src.model.net import ChromoformerClassifier
from src.utils.tools import seed_everything
from src.utils.constants import DEVICE
from src.model.constants import get_config, MARKS, PERTURBATION_STRENGTH, PERTURBATION_REGION

torch.autograd.set_detect_anomaly(False)

# ---------------------- eval helpers ---------------------- #
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def evaluation(out: torch.Tensor, label: torch.Tensor):
    score = out.softmax(dim=1)[:, 1]
    pred = out.argmax(dim=1)
    acc = metrics.accuracy_score(label.numpy(), pred.numpy()) * 100
    auc = metrics.roc_auc_score(label.numpy(), score.numpy()) * 100
    ap  = metrics.average_precision_score(label.numpy(), score.numpy()) * 100
    return score, label, pred, acc, auc, ap

# ---------------------- mask config ---------------------- #
# 运行模式：
# 1) MASK_MODE="all"        -> 跑所有 2^7 个组合（含空集=不mask）
# 2) MASK_MODE="upto_k"     -> 跑从 0..MAX_K 阶的所有组合（例如只跑单/双/三元）
# 3) MASK_MODE="selected"   -> 精确跑 SELECTED_MASKS 中指定的组合
MASK_MODE       = "all"         # "all" | "upto_k" | "selected"
MAX_K           = 3                # 仅当 MASK_MODE="upto_k" 时生效
SELECTED_MASKS  = [
    # 举例：精确指定若干组合（元素必须来自 MARKS）
    # ("H3K27ac",),
    # ("H3K4me3","H3K27ac"),
]
MASK_STRENGTH   = 0.4              # 0=完全抹除；0.5=减半；>1=增强
MASK_REGION     = "all"            # 也可改成 bin 索引列表，如 [10,11,12]
BINSIZES        = [500]

# 其他训练/评测设置
config_path = "configs/default.yaml"
config_base = get_config(config_path)
add_feature_bin = False

remove_marks = None
if remove_marks:
    config_base["remove_marks"] = [remove_marks]
else:
    config_base["remove_marks"] = []

feature_bin_kws = config_base['feature_bin_kws']
seed = config_base["seed"]

config_base['marked_bin_idxes'] = []
config_base['masked_marks'] = {}   # 注意：这里改为 dict（更合理的结构）
bsz = config_base["bsz"]
gamma = config_base["gamma"]

i_max = config_base["i_max"]
w_prom = config_base["w_prom"]
w_max = config_base["w_max"]

n_feats_p = (
    config_base['promoter_feats_basic_nums'] - len(config_base["remove_marks"]) + feature_bin_kws['out_channels']
    if add_feature_bin else
    config_base['promoter_feats_basic_nums'] - len(config_base["remove_marks"])
)
n_feats_pcres = config_base['pcres_feats_basic_nums']
d_emb = config_base["embed"]["d_model"]
embed_kws = config_base["embed"]
pairwise_interaction_kws = config_base["pairwise_interaction"]
regulation_kws = config_base["regulation"]
d_head = config_base["d_head"]
targets = ['bs_label', 'bf_label']
npy_dir = "extra/datasets/processed/v1"

# ---------------------- 生成组合 ---------------------- #
def powerset(iterable):
    """全部子集（含空集）"""
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))

def upto_k(iterable, k):
    """0..k 阶子集"""
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(k + 1))

def build_masked_marks(mask_tuple, strength=0.0, region="all"):
    """把 ('H3K27ac','H3K4me3') -> dict 用于 config['masked_marks']"""
    return {
        m: {
            PERTURBATION_REGION: region,
            PERTURBATION_STRENGTH: float(strength),
        } for m in mask_tuple
    }

if MASK_MODE == "all":
    MASK_SETS = list(powerset(MARKS))
elif MASK_MODE == "upto_k":
    MASK_SETS = list(upto_k(MARKS, MAX_K))
elif MASK_MODE == "selected":
    # 校验
    for tup in SELECTED_MASKS:
        for m in tup:
            assert m in MARKS, f"Unknown mark in SELECTED_MASKS: {m}"
    MASK_SETS = [tuple(t) for t in SELECTED_MASKS]
else:
    raise ValueError("Unknown MASK_MODE")

# ---------------------- 主流程：把 model 加载提速复用 ---------------------- #
predictions = []

for eid in ["E116", "E118", "E003"]:
    for fold in [0, 1, 2, 3]:
        print(f"\n==== eid:{eid}, fold:{fold} ====")

        checkpoints = f"checkpoints/{eid}.{fold}.No_feature_bin.bs_bf_para.model.pt"

        meta_path = f"extra/datasets/processed/v1/meta_datasets/meta_data_{eid}.csv"

        seed_everything(seed)
        meta = (
            pd.read_csv(meta_path).sample(frac=1, random_state=seed).reset_index(drop=True)
        )

        # ——— 构建固定的 train/val 基因划分（对 mask 组合保持一致） ——— #
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
        train_genes = qs[(fold + 0) % 4] + qs[(fold + 1) % 4] + qs[(fold + 2) % 4]
        val_genes   = qs[(fold + 3) % 4]
        print(f"Train/Val gene counts: {len(train_genes)}/{len(val_genes)} | n_feats_p={n_feats_p}")

        # ——— 加载模型（一次） ——— #
        model = ChromoformerClassifier(
            n_feats_p,
            d_emb,
            d_head,
            embed_kws=embed_kws,
            binsizes=BINSIZES,
            seed=42,
            targets=targets,
        ).to(DEVICE)
        ckpt = torch.load(checkpoints, map_location=DEVICE)
        model.load_state_dict(ckpt["net"])
        model.eval()

        # ——— 依次评测不同 mask 组合 ——— #
        for mask_tuple in MASK_SETS:
            # 构造独立 config（不要在原地改）
            config = copy.deepcopy(config_base)
            # 注意：这里使用 dict 结构，兼容你的 Dataset 新实现
            config["masked_marks"] = build_masked_marks(mask_tuple, strength=MASK_STRENGTH, region=MASK_REGION)

            # DataLoader
            val_dataset = BurstformerDataset(
                meta_path,
                npy_dir,
                val_genes,
                i_max,
                BINSIZES,
                w_prom,
                w_max,
                targets=targets,
                config=config,
                with_gene_id=True,
            )
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz)

            # 推理
            bar = tqdm(enumerate(val_loader, 1), total=len(val_loader), leave=False, desc=f"mask={mask_tuple or 'None'}")
            gene_ids, val_out, val_label = [], [], []
            with torch.no_grad():
                for _, d in bar:
                    gene_ids += d.pop('gene_id')
                    for k, v in d.items():
                        if isinstance(v, dict):
                            for _k, _v in v.items():
                                v[_k] = _v.to(DEVICE)
                        else:
                            d[k] = v.to(DEVICE)

                    out = model(
                        d["promoter_feats"][BINSIZES[0]],
                        d["promoter_pad_masks"][BINSIZES[0]],
                    )
                    val_out.append(out.cpu())
                    val_label.append(d["label"].cpu())

            val_out = torch.cat(val_out)
            val_label = torch.cat(val_label)

            # 多任务拆分
            val_preds = {t: p for t, p in zip(targets, torch.chunk(val_out, len(targets), dim=-1))}
            val_labels = {t: y for t, y in zip(targets, torch.chunk(val_label, len(targets), dim=-1))}

            # 逐 target 计算指标并汇总
            records = {"gene_id": gene_ids}
            desc_line = []
            for t in targets:
                s, y, p, acc, auc, ap = evaluation(val_preds[t], val_labels[t])
                records[f'{t}_label'] = y.numpy().squeeze()
                records[f'{t}_score'] = s.numpy().squeeze()
                records[f'{t}_pred']  = p.numpy().squeeze()
                desc_line.append(f"{t}: acc={acc:.3f}, auc={auc:.3f}, ap={ap:.3f}")

            # 附加元信息
            records['fold']         = fold
            records['eid']          = eid
            records['masked_marks'] = "|".join(mask_tuple) if mask_tuple else "None"
            records['mask_strength']= MASK_STRENGTH
            records['mask_region']  = MASK_REGION

            df = pd.DataFrame(records)
            predictions.append(df)
            print(f"[{eid}][fold={fold}] mask={records['masked_marks']} | " + " | ".join(desc_line))

# 汇总所有组合的预测
predictions = pd.concat(predictions, axis=0).reset_index(drop=True)

# 保存（按需要解注释）
out_csv = f"extra/datasets/results/perturbation_masks_mode-{MASK_MODE}-masks_strength-{MASK_STRENGTH}.csv"
predictions.to_csv(out_csv, index=False)
print("Saved to:", out_csv)

if __name__ == "__main__":
    print("Done")