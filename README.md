# **Model Training**

This repository provides the training pipeline for predicting transcriptional burst dynamics using histone modification features and burst labels.

## **Dependencies**

All dependencies are listed in requirements.txt. Install with:

```
pip install -r requirements.txt
```

Key packages include:

- torch – deep learning framework
- numpy, pandas, scipy, scikit-learn – data handling and analysis
- scanpy, anndata – single-cell data utilities
- matplotlib, seaborn – visualization

## Workflow

### **Prepare Data**

Use the preprocessed histone modification features and burst labels generated in the preprocessing step.

### **Training Demo**

Example: training on three cell lines (**E003, E116, E118**) with 4-fold cross-validation:

```
for eid in E003 E116 E118
do
    for fold in 0 1 2 3
    do  
        echo "experiment $eid $fold"
        python train.py \
            --config configs/default.yaml \
            --meta extra/datasets/processed/v1/meta_datasets/meta_data_$eid.csv \
            --npy-dir extra/datasets/processed/v1 \
            --fold $fold \
            -o checkpoints/$eid.$fold.bs_bf_para.model.pt \
            --exp-id 2 \
            --binsizes 500 \
            > logs/$eid.$fold.bs_bf_para.train.log 2>&1
    done
done
```

### **Configuration File**

The training configuration file is a YAML file (e.g. configs/default.yaml) that specifies model, optimization, and biological parameters.

```
seed: 123
# Data Loader 
marks: "H3K4me1\tH3K4me3\tH3K9me3\tH3K27me3\tH3K36me3\tH3K27ac\tH3K9ac"
base_maps: "A:0\tC:1\tG:2\tT:3"

promoter_with_sequence: False  # promoter add sequence statistic feature
pcres_with_sequence: False  # pcres add sequence statistic feature

# Optimization.
num_epoch: 20
lr: 3e-3
bsz: 64
gamma: 0.87
num_works: 8
patience: 20

# Data processing.
i_max: 8
w_prom: 40000
w_max: 40000

# Model specification.
n_feats: 11
feature_bin_kws:
  in_channels: 4
  out_channels: 16
  kernel_size: 6
  dilation: 1
  padding: 'same'
  add_feature_bin: True
embed:
  n_layers: 1
  n_heads: 2
  d_model: 128
  d_ff: 128
pairwise_interaction:
  n_layers: 2
  n_heads: 2
  d_model: 128
  d_ff: 256
  n_feats_pcre: 11
regulation:
  n_layers: 6
  n_heads: 8
  d_model: 256
  d_ff: 256

d_head: 128
```



### **Metadata**

The metadata file is a .csv containing gene-level annotations, expression values, and burst labels.

| **Column name** | **Description**                               |
| --------------- | --------------------------------------------- |
| gene_id         | Gene identifier (e.g., ENSG00000122417)       |
| eid             | Experiment / cell line ID (e.g., E003 for H1) |
| chrom           | Chromosome of gene                            |
| start           | TSS start coordinate (0-based, inclusive)     |
| end             | TSS end coordinate (0-based, exclusive)       |
| strand          | Strand (+ or -)                               |
| bs              | Burst size                                    |
| bf              | Burst frequency                               |
| k_on            | Burst initiation rate                         |
| k_off           | Burst termination rate                        |
| k_syn           | mRNA synthesis rate                           |
| gene_name       | Gene symbol                                   |
| bulk_exp        | Bulk RNA-seq expression level                 |
| mean            | Mean single-cell expression                   |
| bulk_exp_label  | Binary label based on bulk expression         |
| sc_exp_label    | Binary label based on single-cell expression  |
| bs_label        | Binary label for burst size                   |
| bf_label        | Binary label for burst frequency              |
| mean_label      | Binary label for mean expression              |

### **Prediction Demo**

Below is a complete demo showing how to perform random-input prediction using the BurstFormer model.

This demo generates random histone marks' signals , passes them through the model, computes dual-softmax probabilities and evaluates the results.

```
# ==========================================
# BurstFormer Random Prediction Demo (with Mask)
# ==========================================

import torch

from sklearn import metrics
from src.utils.constants import DEVICE
from src.model.constants import get_config
config_path = "configs/default.yaml"
config = get_config(config_path)
# -----------------------------
# 1. Random Input Data
# -----------------------------
batch_size = 64
seq_len = 80
n_feats = 7

# Simulated promoter feature tensor and attention mask
inputs = torch.randn(batch_size, 1, seq_len, n_feats).to(DEVICE)
labels = torch.randint(0, 2, (batch_size, 2)).to(DEVICE)
print(labels.shape)
print(labels[:5])

print("inputs:", inputs.shape)

# -----------------------------
# 2. Load BurstFormer
# -----------------------------
from src.model.net import  BurstFormer

n_feats_p = n_feats
d_head = 128
d_emb = config["embed"]["d_model"]
embed_kws = config["embed"]
d_head = config["d_head"]
binsizes = [500]
targets = ['bs_label','bf_label']

model = BurstFormer(
    n_feats_p,
    d_emb,
    d_head,
    embed_kws=embed_kws,
    binsizes=binsizes,
    seed=42,
    targets=targets,
).to(DEVICE)

# load trained checkpoint, choose model according to cell line and the fold of  the chrom number. here is a demo
# H1 cell line, fold 0
eid = 'E003'
fold = '0'
ckpt = torch.load(f"checkpoints/{eid}.{fold}.bs_bf_para.model.pt", map_location=DEVICE)
model.load_state_dict(ckpt["net"])

model.eval()


def evaluation(out:'torch.Tensor', label:'torch.Tensor'):
    score = out.softmax(axis=1)[:, 1]
    pred = out.argmax(axis=1)

    acc = metrics.accuracy_score(label, pred) * 100
    auc = metrics.roc_auc_score(label, score) * 100
    return score, label, pred, acc, auc


with torch.no_grad():
    val_out = model(inputs)


print("\n=== Output Shapes ===")
print("logits:", val_out.shape)

val_preds = {}
for target, pred in zip(targets,torch.chunk(val_out, len(targets), axis=-1)):
    val_preds[target] = pred

val_labels = {}
for target, label in zip(targets,torch.chunk(labels, len(targets), axis=-1)):
    val_labels[target] = label   
# -----------------------------
# 4. Evaluation
# -----------------------------
    description = "Validation Results: "
for target in targets:
    val_score, val_label, val_pred, val_acc, val_auc = evaluation(val_preds[target],val_labels[target])
    description += f"{target}: acc={val_acc:.4f}, auc={val_auc:.4f}; " 
print(description)
```



## **System Requirements**

BurstFormer was trained on a server equipped with 28 Intel(R) Xeon(R) Gold 6132 CPUs @ 2.60GHz, 60 GB RAM, and 1 NVIDIA A100 GPU with 16 GB memory. 

### **Hardware**

- CPU: ≥ 6 cores, 2.40+ GHz
- GPU: ≥ 16 GB memory
- RAM: ≥ 16 GB

### **Software**

- **OS:** Ubuntu 22.04 (tested)
- **Python:** 3.10.14
- **PyTorch:** 1.13.0 (CUDA 11.2)
- **Other packages:**
  - numpy 1.23.5
  - pandas 2.2.3
  - scikit-learn 1.5.2
  - tqdm 4.66.5
  - scanpy 1.10.3
  - matplotlib 3.9.2

### **Command-line tools (for preprocessing)**

- sambamba 1.0.1
- bedtools 2.31.1