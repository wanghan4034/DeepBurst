import pandas as pd
import torch
from tqdm import tqdm
from sklearn import metrics
from src.utils.tools import seed_everything
from src.utils.constants import DEVICE
from src.model.constants import get_config, MARKS, PERTURBATION_STRENGTH, PERTURBATION_REGION

# 关键：用你的 data.py 对应的 Dataset
# 如果你已经把 data.py 放到了 src/model/data.py，就继续用这一行：
from src.model.data import DeepBurstDataset
# 如果你现在就是用项目根目录下的 data.py（与当前脚本同目录），用这一行：
# from data import DeepBurstDataset

from src.model.net import DeepBurst

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

        # keep 一个 mark，remove 其余 mark（你的原逻辑）
        keep_marks = [keep_mark]
        remove_marks = [m for m in MARKS if m not in keep_marks]
        config["remove_marks"] = remove_marks
        model_tag = "_".join(remove_marks)

        seed = config["seed"]
        bsz = config["bsz"]
        i_max = config["i_max"]
        w_prom = config["w_prom"]
        w_max = config["w_max"]

        # n_feats_p 仍然沿用你原来的写法（注意它必须与 Dataset 输出的 mark 数一致）
        n_feats_p = config["promoter_feats_basic_nums"] - len(remove_marks)

        d_emb = config["embed"]["d_model"]
        embed_kws = config["embed"]
        d_head = config["d_head"]

        targets = ["bs_label", "bf_label"]
        npy_dir = "extra/datasets/processed/v1"
        binsizes = [500]

        # 如果你想指定特定 bins（例如 0..79），在 data.py 中对应 PERTURBATION_REGION
        marked_bin_idxes = list(range(80))  # 可选：也可以用 "all"

        for perturbation_strength in range(11):
            strength = perturbation_strength * 0.1
            config["perturbation_strength"] = strength  # 你原先写了这个字段，保留无妨

            # -----------------------
            # 核心适配：masked_marks 改成 dict
            # -----------------------
            config["masked_marks"] = {
                keep_mark: {
                    PERTURBATION_STRENGTH: strength,
                    # 任选其一：
                    # PERTURBATION_REGION: "all",
                    PERTURBATION_REGION: marked_bin_idxes,
                }
            }

            for fold in [0, 1, 2, 3]:
                print(f"eid:{eid}, fold:{fold}")

                checkpoints = f"checkpoints/{eid}.remove_{model_tag}.{fold}.bs_bf_para.model.pt"
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
                val_genes = qs[(fold + 3) % 4]

                val_dataset = DeepBurstDataset(
                    meta=meta_path,
                    npy_dir=npy_dir,
                    gene_ids=val_genes,
                    i_max=i_max,
                    binsizes=binsizes,
                    w_prom=w_prom,
                    w_max=w_max,
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

                        # data.py 的 item 中除了 promoter_feats，还有 promoter_pad_masks（dict）
                        # 你现有模型 forward 只用 promoter_feats[500]，所以 pad_masks 不影响
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
                desc = ""
                for t in targets:
                    val_score, y_true, y_pred, acc, auc, ap = evaluation(val_preds[t], val_labels[t])
                    records[f"{t}_label"] = y_true.cpu().numpy().squeeze()
                    records[f"{t}_score"] = val_score.cpu().numpy().squeeze()
                    records[f"{t}_pred"]  = y_pred.cpu().numpy().squeeze()
                    desc += f"{t}: acc={acc:.4f}, auc={auc:.4f}, ap={ap:.4f} "

                records["fold"] = fold
                records["keep_mark"] = keep_mark
                records["perturbation_strength"] = strength

                predictions.append(pd.DataFrame(records))
                print(desc)

    predictions = pd.concat(predictions, axis=0)
    predictions.to_csv(f"extra/results/{eid}_perturbation_predictions_bs_bf.csv", index=False)