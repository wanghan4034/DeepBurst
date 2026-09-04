import os
import pandas as pd
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--eid", required=True, help="Cell Type EID.")
parser.add_argument("--delay", required=False, help="Capture efficiency delay.", default=1, type=float)
parser.add_argument("--with_cell_size", action="store_true", help="Enable cell size ratio correction.")
parser.add_argument("--gene_id2neighbors", required=True, help="gene_id2neighbors data file.")
parser.add_argument("--processed_dir", required=False, help="processed_dir", default="extra/datasets/burst/processed")
parser.add_argument("-o", "--output", required=True, help="Output directory.")

args = parser.parse_args()

eid = args.eid
delay = args.delay
with_cell_size = args.with_cell_size
gene_id2neighbors_path = args.gene_id2neighbors
processed_dir = args.processed_dir
saved_dir = args.output


EIDS_TO_CELLTYPES = {
    "E116": "gm12878",
    "E118": "HepG2",
    "E003": "H1",
}

if eid not in EIDS_TO_CELLTYPES:
    raise ValueError(f"Unknown eid: {eid}. Available eids: {list(EIDS_TO_CELLTYPES.keys())}")

os.makedirs(saved_dir, exist_ok=True)


# =========================
# 1. Load data
# =========================

kinetics_path = os.path.join(
    processed_dir,
    f"{EIDS_TO_CELLTYPES[eid]}_statistic_gene_transcript_region_delay_{delay}_with_cellsize_{int(with_cell_size)}.csv",
)

kinetics = pd.read_csv(kinetics_path)

bulk_exp = pd.read_csv(
    "extra/datasets/annotations/exp_raw.tsv",
    sep="\t"
)[["gene_id", eid]]

bulk_exp.columns = ["gene_id", "bulk_exp"]

gene_id2neighbors = pd.read_csv(
    gene_id2neighbors_path,
    names=["gene_id", "eid", "chrom", "start", "end", "strand", "neighbors", "scores"],
)


# =========================
# 2. Merge metadata
# =========================

meta_data = pd.merge(gene_id2neighbors, kinetics, how="inner", on="gene_id")
meta_data = pd.merge(meta_data, bulk_exp, how="inner", on="gene_id")

print(f"[INFO] After merge: {meta_data.shape[0]} genes")


# =========================
# 3. Compute burst kinetic metrics
# =========================
# BS = k_syn / k_off
# BF = k_on * k_off / (k_on + k_off)
# mean = k_on * k_syn / (k_on + k_off)

required_cols = ["k_syn", "k_off", "k_on", "expression", "bulk_exp"]
missing_cols = [col for col in required_cols if col not in meta_data.columns]

if missing_cols:
    raise ValueError(f"Missing required columns in meta_data: {missing_cols}")

eps = 1e-12

meta_data["bs"] = meta_data["k_syn"] / (meta_data["k_off"] + eps)
meta_data["bf"] = meta_data["k_on"] * meta_data["k_off"] / (meta_data["k_on"] + meta_data["k_off"] + eps)
meta_data["mean"] = meta_data["k_on"] * meta_data["k_syn"] / (meta_data["k_on"] + meta_data["k_off"] + eps)

meta_data["cv"] = meta_data.apply(
    lambda row: 0
    if row["mean"] < 0.0001
    else (
        1 / row["mean"]
        + row["k_off"] / (row["k_on"] + eps)
        - ((row["k_on"] + row["k_off"]) / (row["k_on"] + eps))
        * (1 - (row["k_on"] / (1 + row["k_on"])))
        / (((1 + row["k_off"]) / (row["k_off"] + eps)) - row["k_on"] / (1 + row["k_on"]))
    ),
    axis=1,
)

# Remove invalid values
metric_cols = ["bs", "bf", "mean", "cv", "expression", "bulk_exp"]
meta_data = meta_data.replace([np.inf, -np.inf], np.nan)
meta_data = meta_data.dropna(subset=metric_cols).copy()

print(f"[INFO] After removing invalid values: {meta_data.shape[0]} genes")


# =========================
# 4. Keep genes with consistent scRNA-seq and bulk expression labels
# =========================

bulk_exp_median = np.median(meta_data["bulk_exp"])
sc_exp_median = np.median(meta_data["expression"])

meta_data["bulk_exp_label"] = (meta_data["bulk_exp"] > bulk_exp_median).astype(np.int32)
meta_data["sc_exp_label"] = (meta_data["expression"] > sc_exp_median).astype(np.int32)

meta_data = meta_data[meta_data["sc_exp_label"] == meta_data["bulk_exp_label"]].copy()

print(f"[INFO] After sc/bulk expression label consistency filter: {meta_data.shape[0]} genes")


# =========================
# 5. Optional upper-tail filtering
# =========================
# Keep the original logic:
# remove genes with extremely high bs and bf unless mean > 1.

bs_upper_threshold = np.percentile(meta_data["bs"], 95)
bf_upper_threshold = np.percentile(meta_data["bf"], 95)
mean_threshold = 1

before_upper_filter = meta_data.shape[0]

meta_data = meta_data[
    ((meta_data["bs"] < bs_upper_threshold) & (meta_data["bf"] < bf_upper_threshold))
    | (meta_data["mean"] > mean_threshold)
].copy()

print(
    f"[INFO] After upper-tail filter: {meta_data.shape[0]} genes "
    f"(removed {before_upper_filter - meta_data.shape[0]})"
)


# =========================
# 6. Median thresholds and exclude genes within ±10 percentile around median
# =========================
# Here, "median ±10%" is interpreted as the 40th-60th percentile interval.
# Labels are still defined by the median threshold computed before removing ambiguous genes.

bs_median = np.median(meta_data["bs"])
bf_median = np.median(meta_data["bf"])
cv_median = np.median(meta_data["cv"])
mean_median = np.median(meta_data["mean"])

bs_low, bs_high = np.percentile(meta_data["bs"], [40, 60])
bf_low, bf_high = np.percentile(meta_data["bf"], [40, 60])

print(f"[INFO] BS median: {bs_median:.6g}")
print(f"[INFO] BF median: {bf_median:.6g}")
print(f"[INFO] BS ambiguous interval [40%, 60%]: [{bs_low:.6g}, {bs_high:.6g}]")
print(f"[INFO] BF ambiguous interval [40%, 60%]: [{bf_low:.6g}, {bf_high:.6g}]")

# Assign labels using median thresholds before removing ambiguous genes
meta_data["bs_label"] = (meta_data["bs"] > bs_median).astype(np.int32)
meta_data["bf_label"] = (meta_data["bf"] > bf_median).astype(np.int32)
meta_data["cv_label"] = (meta_data["cv"] > cv_median).astype(np.int32)
meta_data["mean_label"] = (meta_data["mean"] > mean_median).astype(np.int32)

before_middle_filter = meta_data.shape[0]

# Remove genes close to either BS median or BF median
meta_data = meta_data[
    (~meta_data["bs"].between(bs_low, bs_high, inclusive="both"))
    & (~meta_data["bf"].between(bf_low, bf_high, inclusive="both"))
].copy()

print(
    f"[INFO] After removing genes within BS/BF median ±10 percentile intervals: "
    f"{meta_data.shape[0]} genes "
    f"(removed {before_middle_filter - meta_data.shape[0]})"
)

print("[INFO] BS label counts after filtering:")
print(meta_data["bs_label"].value_counts().sort_index())

print("[INFO] BF label counts after filtering:")
print(meta_data["bf_label"].value_counts().sort_index())


# =========================
# 7. Save
# =========================

output_path = os.path.join(
    saved_dir,
    f"meta_data_{eid}_delay_{delay}_with_cellsize_{int(with_cell_size)}_threshold_median_exclude_mid20.csv",
)

meta_data.to_csv(output_path, index=False)

print(f"[DONE] Saved to: {output_path}")