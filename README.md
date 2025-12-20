
# **Model overview**

DeepBurst is a transformer-based framework that predicts genome-wide transcriptional bursting kinetics from promoter-proximal histone modification profiles. It aims to quantify how chromatin states shape bursting behavior by jointly modeling burst frequency and burst size.

For each gene, we represent the local regulatory context using a 40 kb window centered at the transcription start site. ChIP–seq coverage tracks for seven canonical histone modifications spanning activating and repressive chromatin states (H3K9ac, H3K27ac, H3K4me3, H3K4me1, H3K36me3, H3K27me3, H3K9me3) are aggregated into 80 bins of 500 bp. A Transformer encoder then learns dependencies across genomic positions and across histone marks, producing a position-resolved regulatory embedding.

To generate supervisory labels, gene-specific bursting parameters are inferred from matched single-cell RNA-seq data using stochastic transcription models and aligned with the corresponding histone profiles in the same cell line. Because capture efficiency, cell type, and inference pipelines can introduce systematic shifts in continuous kinetic estimates, we formulate prediction as a binary classification task rather than regression. Within each cell line, genes are labeled as high or low burst frequency and high or low burst size using median-based thresholds, yielding robust labels for training.

The transcription-start-site–centered embedding extracted from the Transformer output is passed through fully connected layers and two parallel binary classifiers to predict burst frequency and burst size labels. Model training and evaluation are performed separately per cell type using fourfold chromosome-split cross-validation to prevent information leakage between training and validation sets.

Once trained, the model supports downstream analyses including genome-wide bursting-state prediction from histone profiles, identification of informative histone marks and genomic positions, and in silico perturbation analyses that prioritize histone-signal changes predicted to shift genes between bursting states.

# **Model training**

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

## Training  and prediction

### **Prepare data**

Use the preprocessed histone modification features and burst labels generated in the preprocessing step.

### **Training demo**

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

### **Prediction Demo**

Below is a complete demo showing how to perform random-input prediction using the DeepBurst model.

This demo generates random histone marks' signals , passes them through the model, computes dual-softmax probabilities and evaluates the results.

```
# ==========================================
# DeepBurst Random Prediction Demo (with Mask)
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
# 2. Load DeepBurst
# -----------------------------
from src.model.net import  DeepBurst

n_feats_p = n_feats
d_head = 128
d_emb = config["embed"]["d_model"]
embed_kws = config["embed"]
d_head = config["d_head"]
binsizes = [500]
targets = ['bs_label','bf_label']

model = DeepBurst(
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



# **Design**

This script provides a minimal, end-to-end demo of **DeepBurst** inference and evaluation using synthetic inputs. It is intended as a **sanity check** for model loading, forward pass, output parsing, and metric computation.

1. **Synthetic batch construction**

   The script creates a simulated promoter-feature tensor inputs with shape (batch_size, 1, seq_len, n_feats) and corresponding binary labels labels with shape (batch_size, 2), where the two label columns align with the two prediction targets.

2. **Model instantiation and checkpoint loading**

   A DeepBurst model is instantiated using hyperparameters from configs/default.yaml. A trained checkpoint is then loaded based on the selected **cell line ID (**eid**)** and **chromosome-fold (**fold**)** (e.g., E003 fold 0). The model is switched to evaluation mode to ensure deterministic inference behavior.

3. **Multi-target output handling**

   The model outputs a single tensor of logits that concatenates predictions for multiple targets along the last dimension. The script splits this tensor into per-target logits using torch.chunk, ensuring a one-to-one mapping between:

   - targets = ['bs_label', 'bf_label']
   - per-target predictions (val_preds[target])
   - per-target labels (val_labels[target])

4. **Evaluation metrics**

   For each target, the script computes:

   - **ROC-AUC** based on the positive-class probability from softmax

     Metrics are reported independently for each target to make it easy to verify that the output format and downstream evaluation pipeline are correct.

# **System Requirements**

## **Training DeepBurst**

DeepBurst is trained on a server equipped with **28 Intel(R) Xeon(R) Gold 6132 CPUs @ 2.60GHz**, **60 GB RAM**, and **one NVIDIA V100 GPU (16 GB VRAM)**. With sufficient GPU memory, training is also feasible on modern consumer GPUs such as the **RTX 4090**.

### **Hardware**

- **CPU:** ≥ 6 cores, ≥ 2.40 GHz
- **GPU:** ≥ 16 GB VRAM
- **RAM:** ≥ 16 GB

### **Software**

- **OS:** Ubuntu 22.04 (tested)
- **Python:** 3.10.14
- **PyTorch:** 1.13.0 (CUDA 11.2)
- **Other Key Python packages are listed in requirements.txt**

### **Command-line tools (for preprocessing)**

- sambamba 1.0.1
- bedtools 2.31.1

## **Design / Enformer-based inference**

During the design and analysis stage, we run **Enformer** to generate sequence-based predictions using **DeepMind’s official TensorFlow implementation** and the **official pre-trained model checkpoints**. Inference in this project is performed on an **NVIDIA A800 GPU (80 GB VRAM)**. Once histone modification features are prepared, running **DeepBurst** inference follows the same hardware and software configuration described above.