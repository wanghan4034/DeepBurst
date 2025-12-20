# **Data Preprocessing**

This repository provides preprocessing scripts for constructing model-ready datasets for transcriptional burst prediction. The pipeline integrates **histone modification features** with **bursting labels** (burst frequency and burst size, including expression and noise), and produces standardized inputs for downstream training and evaluation.

The examples below demonstrate the workflow for the **H1 (E003) cell line**.

## **Dependencies**

### **For histone feature extraction and burst-label preprocessing (Python)**

- Python ≥ 3.10
- Core packages:

```
pip install numpy pandas scipy scikit-learn
```

### **For DeepTX-based kinetic inference (Julia)**

- Julia ≥ 1.7.3
- Required Julia packages:

```
Pkg.add([
  "Flux",
  "CSV",
  "DataFrames",
  "JLD2",
  "Distributions",
  "StatsBase",
  "SpecialFunctions",
  "ZygoteRules",
  "Catalyst",
  "BlackBoxOptim",
  "Sobol",
  "Distances",
  "OptimalTransport",
  "ProgressMeter",
  "MAT",
  "Interp1d"
])
```



## **Preprocessing Pipeline**

### **1. Histone Modification Feature Generation**

Histone modification signals are extracted from epigenomic tracks and aggregated into per-gene feature matrices.

**Command (H1 / E003):**

```
python src/data/data_process.py \
  --eid E003 \
  --gene extra/datasets/genomic/hg19/genes.bed \
  --epi_dir extra/datasets/epigenetic/hg19 \
  -o extra/datasets/processed/v2
```

**Arguments**

- --eid: Experiment / cell line ID (e.g., E003 for H1)
- --gene: Gene annotation file in BED format
- --epi_dir: Directory containing histone modification tracks
- -o: Output directory



**Outputs**

- Per-gene histone feature matrices (e.g., .csv / .npy, depending on configuration)

### **2. Burst Label Generation**

Burst frequency and burst size are inferred from UMI count data (scRNA-data), and then converted into binary labels.

#### **Step 1 — Infer bursting kinetics**

**txburst**

```
python src/data_preprocessing/label_generation/txburst/txburst_infer.py --cell_type H1
```

- Output: **continuous** burst kinetics parameters in .csv format

**DeepTX**

```
julia TX_inferrer.jl data/H1_scRNA.csv inferred_results.csv
```

- Output: **continuous** burst kinetics parameters in .csv format

#### **Step 2 — Convert kinetics into burst labels**

```
python src/data/burst/data_convert.py \
  --eid E003 \
  --gene_id2neighbors extra/datasets/processed/v1/E003/gene_id2neighbors_E003.csv \
  -o extra/datasets/processed/v1/meta_datasets
```

**Outputs**

- Binary burst labels and related quantities per gene (e.g., BF/BS labels, expression/noise labels) in meta_data.csv



### **3. Final Outputs**

After preprocessing, the pipeline produces:

- **Histone modification features**: per-gene profiles across histone marks
- **Burst labels**: per-gene BF/BS (and optional expression/noise) labels

