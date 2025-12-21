# **Model overview**

DeepBurst is a Transformer-based framework for predicting genome-wide transcriptional bursting kinetics from promoter-proximal histone modification profiles. The model is designed to quantify how chromatin states shape bursting behavior by jointly modeling **burst frequency** and **burst size**.

For each gene, the local regulatory context is represented by a **40 kb window** centered on the transcription start site (TSS). ChIP–seq coverage tracks for seven canonical histone modifications spanning activating and repressive chromatin states (**H3K9ac, H3K27ac, H3K4me3, H3K4me1, H3K36me3, H3K27me3, H3K9me3**) are aggregated into **80 bins of 500 bp**. A Transformer encoder learns dependencies across genomic positions and across histone marks, producing a position-resolved regulatory embedding.

Supervisory labels are derived by inferring gene-specific bursting parameters from matched single-cell RNA-seq data using stochastic transcription models, and aligning these estimates with histone profiles from the same cell line. Because capture efficiency, cell type, and inference pipelines can introduce systematic shifts in continuous kinetic estimates, DeepBurst is trained as a **binary classification** model rather than a regression model. Within each cell line, genes are labeled as **high/low burst frequency** and **high/low burst size** using **median-based thresholds**, yielding robust training targets.

The TSS-centered embedding extracted from the Transformer output is passed through fully connected layers and two parallel binary classifiers to predict burst frequency and burst size labels. Model training and evaluation are performed **separately per cell type** using **four-fold chromosome-split cross-validation** to reduce the risk of information leakage between training and validation sets.

Once trained, DeepBurst supports downstream analyses including genome-wide bursting-state prediction from histone profiles, identification of informative histone marks and genomic positions, and **in silico perturbation** analyses that prioritize histone-signal changes predicted to shift genes between bursting states.

# **Data preprocessing**

This repository provides preprocessing scripts for constructing model-ready datasets for transcriptional burst prediction. The pipeline integrates **histone modification features** with **bursting labels** (burst frequency and burst size, optionally including expression and noise), and produces standardized inputs for downstream training and evaluation.

Cell lines are indexed using Roadmap Epigenomics **EID** identifiers defined in the **Roadmap 2015** reference epigenome compendium. In this repository, **E003** corresponds to **H1**, **E118** corresponds to **HepG2**, and **E116** corresponds to **GM12878**.



## **Dependencies**

### **For histone feature extraction and burst-label preprocessing (Python)**

python3.10, All dependencies are listed in requirements.txt. Install with:

```
pip install -r requirements.txt
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



## **Histone modification feature generation**

<img src="img/data_processing.png" alt="data_processing" style="zoom:80%;" />

### **1. Download and preprocessing**

We downloaded the raw **TagAlign** files from the specified [data source. We then used **bedtools** to convert TagAlign to **BAM**, and used **sambamba** to sort and index the BAM files. Next, we ran **bedtools genomecov** to compute base-level genome-wide coverage in the **hg19** reference coordinate system, and exported the results as **bigWig** files for downstream feature construction.

This workflow is encapsulated in src/data_preprocessing/Snakefile and can be executed with:

```
snakemake -s src/data_preprocessing/Snakefile -j 8
```



### **2. Histone modification feature extraction**

The example below demonstrates the workflow for the **H1 (E003)** cell line. Histone modification signals are extracted from epigenomic tracks and aggregated into per-gene feature matrices.

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



## **Burst label generation**

Burst frequency and burst size are inferred from UMI counts (scRNA-seq) and then converted into binary labels.

<img src="img/label_generation.png" alt="label_generation" style="zoom:100%;" />

### **Step 1 — Infer bursting kinetics**

**txburst**

```
python src/data_preprocessing/label_generation/txburst/txburst_infer.py --cell_type H1
```

- Output: **continuous** bursting parameters in .csv

**DeepTX**

```
julia TX_inferrer.jl data/H1_scRNA.csv inferred_results.csv
```



- Output: **continuous** bursting parameters in .csv

### **Step 2 — Convert kinetics into burst labels**

```
python src/data/burst/data_convert.py \
  --eid E003 \
  --gene_id2neighbors extra/datasets/processed/v1/E003/gene_id2neighbors_E003.csv \
  -o extra/datasets/processed/v1/meta_datasets
```

**Outputs**

- Per-gene binary burst labels and related quantities (e.g., BF/BS labels; optional expression/noise labels) saved as meta_data_*.csv



## **Final outputs**

After preprocessing, the pipeline produces:

- **Histone modification features**: per-gene profiles across histone marks
- **Burst labels**: per-gene BF/BS labels (optionally expression/noise labels)



# **Model training and prediction**

This repository provides a training pipeline for predicting transcriptional burst dynamics from histone modification features and burst-related labels.

## **Dependencies**

All dependencies are listed in requirements.txt. Install them with:

```
pip install -r requirements.txt
```



## **Training and prediction**

### **Training demo**

Example: training on three Roadmap cell lines (**E003, E116, E118**) with 4-fold cross-validation:

```
for eid in E003 E116 E118
do
    for fold in 0 1 2 3
    do
        echo "experiment $eid $fold"
        python train.py \
            --config configs/default.yaml \
            --meta extra/datasets/processed/v1/meta_datasets/meta_data_${eid}.csv \
            --npy-dir extra/datasets/processed/v1 \
            --fold $fold \
            -o checkpoints/${eid}.${fold}.bs_bf_para.model.pt \
            --exp-id 2 \
            --binsizes 500 \
            > logs/${eid}.${fold}.bs_bf_para.train.log 2>&1
    done
done
```



### **Prediction demo (random inputs)**

Below is a minimal, end-to-end example showing how to run inference with **DeepBurst** using synthetic inputs. The script (i) generates random histone-mark signals, (ii) performs a forward pass, (iii) converts logits to probabilities, and (iv) extracts per-target predictions.

Import required packages and load the configuration:

```
import torch
from sklearn import metrics

from src.utils.constants import DEVICE
from src.model.constants import get_config

config_path = "configs/default.yaml"
config = get_config(config_path)
```

Create synthetic histone-mark inputs:

```
# -----------------------------
# 1. Random Input Data
# -----------------------------
batch_size = 64
seq_len = 80
n_feats = 7  # number of histone marks / features

inputs = torch.randn(batch_size, 1, seq_len, n_feats).to(DEVICE)
labels = torch.randint(0, 2, (batch_size, 2)).to(DEVICE)

print("labels:", labels.shape)
print(labels[:5])
print("inputs:", inputs.shape)
```

Instantiate the model and load a trained checkpoint:

```
# -----------------------------
# 2. Load DeepBurst
# -----------------------------
from src.model.net import DeepBurst

binsizes = [500]
targets = ["bs_label", "bf_label"]

d_emb = config["embed"]["d_model"]
embed_kws = config["embed"]
d_head = config["d_head"]

model = DeepBurst(
    n_feats,
    d_emb,
    d_head,
    embed_kws=embed_kws,
    binsizes=binsizes,
    seed=42,
    targets=targets,
).to(DEVICE)

# Example: load a checkpoint for a specific cell line and fold
eid = "E003"
fold = "0"
ckpt = torch.load(f"checkpoints/{eid}.{fold}.bs_bf_para.model.pt", map_location=DEVICE)
model.load_state_dict(ckpt["net"])
```

Run prediction and parse outputs:

```
model.eval()
with torch.no_grad():
    out = model(inputs)

print("\n=== Output Shapes ===")
print("logits:", out.shape)

# Split logits into per-target blocks
for target, target_out in zip(targets, torch.chunk(out, len(targets), dim=-1)):
    prob_pos = target_out.softmax(dim=1)[:, 1]   # P(class=1)
    pred = target_out.argmax(dim=1)              # predicted class (0/1)
    print(f"{target}: prob_pos={prob_pos[:5]}, pred={pred[:5]}")
```



### **Additional scripts**

More analysis scripts (e.g., cell-type–specific vs. cell-type–agnostic inference, cross-cell-type evaluation, single-mark prediction, and in silico perturbation) are located under scripts/analysis/. All plotting scripts used in the manuscript are under scripts/figures/.



# **Design notes (inference sanity check)**

<img src="img/design_module.png" alt="design_module" style="zoom:100%;" />

This demo provides a minimal end-to-end sanity check for **DeepBurst** inference and evaluation using synthetic inputs. It is intended to validate checkpoint loading, forward pass execution, output parsing, and metric computation.

1. **Synthetic batch construction**

   The demo creates a simulated promoter-feature tensor inputs with shape (batch_size, 1, seq_len, n_feats) and corresponding binary labels with shape (batch_size, 2), where the two label columns align with the two prediction targets.

2. **Model instantiation and checkpoint loading**

   The DeepBurst model is instantiated using hyperparameters from configs/default.yaml. A trained checkpoint is loaded based on the selected **cell line ID (****eid****)** and **chromosome fold (****fold****)** (e.g., E003, fold 0). The model is switched to evaluation mode to ensure deterministic inference behavior.

3. **Multi-target output handling**

   The model outputs a single tensor of logits that concatenates predictions for multiple targets along the last dimension. The script splits this tensor into per-target logits using torch.chunk, maintaining a one-to-one mapping between:

   - targets = ["bs_label", "bf_label"]
   - per-target predictions
   - per-target labels

4. **Evaluation metrics**

   For each target, the demo computes:

   - **ROC-AUC**, using the positive-class probability from softmax

Metrics are reported independently for each target to confirm that the output format and evaluation pipeline are consistent.

# **System requirements**

## **Training DeepBurst**

DeepBurst is trained on a server equipped with **28 Intel(R) Xeon(R) Gold 6132 CPUs @ 2.60GHz**, **60 GB RAM**, and **one NVIDIA V100 GPU (16 GB VRAM)**. With sufficient GPU memory, training is also feasible on modern consumer GPUs (e.g., **RTX 4090**).

### **Hardware**

- **CPU:** ≥ 6 cores, ≥ 2.40 GHz
- **GPU:** ≥ 16 GB VRAM
- **RAM:** ≥ 16 GB

### **Software**

- **OS:** Ubuntu 22.04 (tested)
- **Python:** 3.10.14
- **PyTorch:** 1.13.0 (CUDA 11.2)
- Other key Python packages are listed in requirements.txt

### **Command-line tools (for preprocessing)**

- sambamba 1.0.1
- bedtools 2.31.1





## **Design stage (Enformer-based inference)**

During the design and analysis stage, we run **Enformer** to generate sequence-based predictions using DeepMind’s official TensorFlow implementation and the official pre-trained checkpoints. Inference is performed on an **NVIDIA A800 GPU (80 GB VRAM)**. Once histone modification features are prepared, running **DeepBurst** inference uses the same hardware and software configuration described above.