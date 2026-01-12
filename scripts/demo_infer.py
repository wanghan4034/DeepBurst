import argparse
from pathlib import Path

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


def evaluation(out: torch.Tensor, label: torch.Tensor):
    """Return score/pred and common binary classification metrics."""
    score = out.softmax(dim=1)[:, 1]
    pred = out.argmax(dim=1)

    acc = metrics.accuracy_score(label, pred) * 100
    auc = metrics.roc_auc_score(label, score) * 100
    ap = metrics.average_precision_score(label, score) * 100
    return score, label, pred, acc, auc, ap


def parse_args():
    p = argparse.ArgumentParser("DeepBurst demo inference")

    # Required
    p.add_argument("--eid", type=str, required=True, help="Cell line EID, e.g. E003")
    p.add_argument("--ckpt", type=str, required=True, help="Checkpoint path, e.g. checkpoints/E003.0.bs_bf_para.model.pt")
    p.add_argument("--npy_dir", type=str, required=True, help="Processed dataset root, e.g. extra/datasets/processed/v1")

    # Optional: meta path. If not provided, we auto-build it from eid + npy_dir.
    p.add_argument(
        "--meta",
        type=str,
        default=None,
        help="Optional meta CSV path. If omitted, use {npy_dir}/meta_datasets/meta_data_{eid}.csv",
    )

    p.add_argument("--config", type=str, default="configs/default.yaml", help="YAML config path")
    p.add_argument("--out", type=str, default="extra/results/infer_demo.csv", help="Output CSV path")

    # Compatibility knobs
    p.add_argument("--remove_marks", type=str, default=None, help="Optional single mark name to remove (compat)")
    p.add_argument("--fold", type=int, default=0, help="Fold index used to define validation split (default: 0)")

    return p.parse_args()


def build_chromosome_splits():
    """Chromosome-based 4-fold mapping (same as your training split)."""
    splits = {
        1: ["chr1", "chr6", "chr5", "chr8", "chr14", "chrY"],
        2: ["chr7", "chr10", "chr11", "chr12", "chr15", "chr21"],
        3: ["chr2", "chr3", "chr4", "chr16", "chr18", "chr20"],
        4: ["chr9", "chr13", "chr17", "chr19", "chr22", "chrX"],
    }
    return {chrom: k for k, chroms in splits.items() for chrom in chroms}


def resolve_meta_path(eid: str, npy_dir: str, meta_arg: str | None) -> str:
    """Resolve meta_path: use --meta if provided, else build default from npy_dir + eid."""
    if meta_arg is not None and str(meta_arg).strip() != "":
        return meta_arg

    # Default: {npy_dir}/meta_datasets/meta_data_{eid}.csv
    default_meta = Path(npy_dir) / "meta_datasets" / f"meta_data_{eid}.csv"
    if not default_meta.exists():
        raise FileNotFoundError(
            f"Meta file not found.\n"
            f"  Tried default: {default_meta}\n"
            f"  You can pass an explicit path via: --meta /path/to/meta.csv"
        )
    return str(default_meta)


def main():
    args = parse_args()

    # ----- Resolve meta path (supports auto path building) -----
    meta_path = resolve_meta_path(args.eid, args.npy_dir, args.meta)

    # ----- Load config -----
    config = get_config(args.config)
    config["remove_marks"] = [args.remove_marks] if args.remove_marks else []

    seed_everything(config["seed"])
    config["marked_bin_idxes"] = []
    config["masked_marks"] = []

    # ----- Constants from config -----
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

    # ----- Load meta and build chromosome-based split (vectorized) -----
    meta = pd.read_csv(meta_path).sample(frac=1, random_state=config["seed"]).reset_index(drop=True)

    chromosome_splits = build_chromosome_splits()
    split_id = meta["chrom"].map(chromosome_splits)  # unknown chrom -> NaN

    # Filter out rows with unknown chromosomes to avoid KeyError
    meta = meta.loc[split_id.notna()].copy()
    split_id = split_id.loc[meta.index].astype(int)


    val_genes = list(meta['gene_id'])

    print(f"eid={args.eid}, fold={args.fold}")
    print(f"ckpt={args.ckpt}")
    print(f"meta={meta_path}")
    print(f"npy_dir={args.npy_dir}")
    print(f"val={len(val_genes)}, n_feats_p={n_feats_p}")

    # ----- Dataset / Loader -----
    val_dataset = DeepBurstDataset(
        meta_path,
        args.npy_dir,
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

    # ----- Model -----
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

    # ----- Evaluation -----
    bar = tqdm(val_loader, total=len(val_loader))
    gene_ids, val_out, val_label = [], [], []

    model.eval()
    with torch.no_grad():
        for d in bar:
            gene_ids += d.pop("gene_id")

            # Move batch to device
            for k, v in d.items():
                if isinstance(v, dict):
                    for _k, _v in v.items():
                        v[_k] = _v.to(DEVICE)
                else:
                    d[k] = v.to(DEVICE)

            out = model(
                d["promoter_feats"][500],            )
            val_out.append(out.cpu())
            val_label.append(d["label"].cpu())

    val_out = torch.cat(val_out, dim=0)
    val_label = torch.cat(val_label, dim=0)

    val_preds = dict(zip(targets, torch.chunk(val_out, len(targets), dim=-1)))
    val_labels = dict(zip(targets, torch.chunk(val_label, len(targets), dim=-1)))

    # ----- Metrics + Save -----
    records = {
        "gene_id": gene_ids,
        "eid": args.eid,
        "fold": args.fold,
        "ckpt": args.ckpt,
        "meta": meta_path,
        "npy_dir": args.npy_dir,
    }

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