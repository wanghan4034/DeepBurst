import argparse
import torch
import pandas as pd
from tqdm import tqdm
from sklearn import metrics

from src.model.data import DeepBurstDataset
from src.model.net import DeepBurst
from src.utils.tools import seed_everything
from src.utils.constants import DEVICE
from src.model.constants import get_config

torch.autograd.set_detect_anomaly(True)


def evaluation(out: "torch.Tensor", label: "torch.Tensor"):
    score = out.softmax(axis=1)[:, 1]
    pred = out.argmax(axis=1)

    acc = metrics.accuracy_score(label, pred) * 100
    auc = metrics.roc_auc_score(label, score) * 100
    ap = metrics.average_precision_score(label, score) * 100
    return score, label, pred, acc, auc, ap


def parse_args():
    p = argparse.ArgumentParser("DeepBurst demo inference")
    p.add_argument("--eid", type=str, required=True, help="Cell line EID, e.g. E003")
    p.add_argument("--ckpt", type=str, required=True, help="Checkpoint path, e.g. checkpoints/E003.0.bs_bf_para.model.pt")
    p.add_argument("--data", type=str, required=True, help="Processed dataset root, e.g. extra/datasets/processed/v1")
    p.add_argument("--config", type=str, default="configs/default.yaml", help="YAML config path")
    p.add_argument("--out", type=str, default="extra/results/infer_demo.csv", help="Output CSV path")
    p.add_argument("--remove_marks", type=str, default=None, help="Optional single mark name to remove (compat)")
    p.add_argument("--fold", type=int, default=None, help="Optional: record fold in output CSV (does not affect split unless provided)")
    return p.parse_args()


def main():
    args = parse_args()

    # ---- config ----
    config = get_config(args.config)

    if args.remove_marks:
        config["remove_marks"] = [args.remove_marks]
    else:
        config["remove_marks"] = []

    seed = config["seed"]
    seed_everything(seed)

    config["marked_bin_idxes"] = []
    config["masked_marks"] = []

    # ---- constants from config ----
    bsz = config["bsz"]
    i_max = config["i_max"]
    w_prom = config["w_prom"]
    w_max = config["w_max"]

    n_feats_p = config["promoter_feats_basic_nums"] - len(config["remove_marks"])
    d_emb = config["embed"]["d_model"]
    embed_kws = config["embed"]
    d_head = config["d_head"]

    targets = ["bs_label", "bf_label"]
    binsizes = [500]

    # ---- paths ----
    npy_dir = args.data
    meta_path = f"{npy_dir}/meta_datasets/meta_data_{args.eid}.csv"

    # ---- load meta + build chromosome CV split pool (same as your original) ----
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

    # 你现在只传一个 ckpt；这里需要一个 fold 来定义 val 集合。
    # 若不传 --fold，则默认按 fold=0（对应你之前循环的 fold=0）。
    fold = 0 if args.fold is None else int(args.fold)

    train_genes = qs[(fold + 0) % 4] + qs[(fold + 1) % 4] + qs[(fold + 2) % 4]
    val_genes = qs[(fold + 3) % 4]

    print(f"eid={args.eid}, ckpt={args.ckpt}, data={args.data}, fold={fold}")
    print(f"train={len(train_genes)}, val={len(val_genes)}, n_feats_p={n_feats_p}")

    # ---- dataset/loader ----
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

    # ---- model ----
    model = DeepBurst(
        n_feats_p,
        d_emb,
        d_head,
        embed_kws=embed_kws,
        binsizes=binsizes,
        seed=42,
        targets=targets,
    ).to(DEVICE)

    ckpt = torch.load(args.ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt["net"])

    # ---- eval ----
    bar = tqdm(enumerate(val_loader, 1), total=len(val_loader))
    gene_ids, val_out, val_label = [], [], []

    model.eval()
    with torch.no_grad():
        for _, d in bar:
            gene_ids += d.pop("gene_id")
            for k, v in d.items():
                if isinstance(v, dict):
                    for _k, _v in v.items():
                        v[_k] = _v.to(DEVICE)
                else:
                    d[k] = v.to(DEVICE)

            out = model(
                d["promoter_feats"][500],
                d["promoter_pad_masks"][500],
            )
            val_out.append(out.cpu())
            val_label.append(d["label"].cpu())

    val_out = torch.cat(val_out)
    val_label = torch.cat(val_label)

    val_preds = {t: p for t, p in zip(targets, torch.chunk(val_out, len(targets), axis=-1))}
    val_labels = {t: l for t, l in zip(targets, torch.chunk(val_label, len(targets), axis=-1))}

    # ---- metrics + save ----
    records = {"gene_id": gene_ids, "eid": args.eid, "fold": fold, "ckpt": args.ckpt}
    description = ""
    for target in targets:
        val_score, val_lab, val_pred, val_acc, val_auc, val_ap = evaluation(val_preds[target], val_labels[target])
        records[f"{target}_label"] = val_lab.cpu().numpy().squeeze()
        records[f"{target}_score"] = val_score.cpu().numpy().squeeze()
        records[f"{target}_pred"] = val_pred.cpu().numpy().squeeze()
        description += f"{target}: acc={val_acc:.4f}, auc={val_auc:.4f}, ap={val_ap:.4f}  "

    df = pd.DataFrame(records)
    df.to_csv(args.out, index=False)
    print(description)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()