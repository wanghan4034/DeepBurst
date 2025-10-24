

# **Data Preprocessing**

This repository provides preprocessing scripts for building datasets used in transcriptional burst prediction. The pipeline integrates **histone modification features** and **bursting labels** into model-ready inputs.

Below we show examples for the **H1 (E003) cell line**.

## **Dependencies**

- Python ≥ 3.8
- Required packages:



```
pip install numpy pandas scipy scikit-learn
```

## Preprocessing  Pipeline

### **1. Histone Modification Feature Generation**

Histone modification signals are extracted and aggregated into per-gene feature matrices.

**Command (H1 / E003):**

```
python src/data/data_process.py \
    --eid E003 \
    --gene extra/datasets/genomic/hg19/genes.bed \
    --epi_dir extra/datasets/epigenetic/hg19 \
    -o extra/datasets/processed/v2
```

- **Arguments:**
  - --eid : Experiment / cell line ID (E003 for H1)
  - --gene : Gene annotation file (genes.bed)
  - --epi_dir : Directory with histone modification tracks
  - -o : Output directory
- **Outputs:**
  - Histone feature matrix (.csv / .npy)

### **2. Burst Label Generation**

Burst frequency (BF) and burst size (BS) are computed from UMI counts.

**Step 1 – Extract UMI counts:**

```
nohup python src/data/burst/extract_umi_counts.py --cell_type H1 \
    > logs/E003.extract_umi_counts.log 2>&1 &
```

- Output: UMI count matrix (.csv)

**Step 2 – Convert to burst labels:**

```
python src/data/burst/data_convert.py \
    --eid E003 \
    --gene_id2neighbors extra/datasets/processed/v1/E003/gene_id2neighbors_E003.csv \
    -o extra/datasets/processed/v1/meta_datasets
```

- Output: Burst labels (BF, BS) in extra/datasets/processed/v1/meta_datasets/

### **3. Final Outputs**

- **Histone modification features**: per-gene histone mark profiles
- **Burst labels**: burst frequency & burst size per gene