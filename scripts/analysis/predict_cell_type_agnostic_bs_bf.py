import argparse
import torch
import torch.nn as nn
import os
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import seaborn as sns
from src.model.data import DeepBurstDataset
from src.model.net import  DeepBurst
from src.utils.tools import seed_everything
from src.utils.constants import DEVICE
from src.model.constants import get_config


torch.autograd.set_detect_anomaly(True)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

#
# Training setup.
#
    
def evaluation(out:'torch.Tensor', label:'torch.Tensor'):
    score = out.softmax(axis=1)[:, 1]
    pred = out.argmax(axis=1)

    acc = metrics.accuracy_score(label, pred) * 100
    auc = metrics.roc_auc_score(label, score) * 100
    ap = metrics.average_precision_score(label, score) * 100
    return score, label, pred, acc, auc, ap

config_path = "configs/default.yaml"
config = get_config(config_path)


config["remove_marks"] = []


seed = config["seed"]

config['marked_bin_idxes'] = []
config['masked_marks'] = []

bsz = config["bsz"]
gamma = config["gamma"]

i_max = config["i_max"]
w_prom = config["w_prom"]
w_max = config["w_max"]

n_feats_p = config['promoter_feats_basic_nums'] - len(config["remove_marks"])
d_emb = config["embed"]["d_model"]
embed_kws = config["embed"]
d_head = config["d_head"]
targets = ['bs_label','bf_label']
npy_dir = "/Volumes/ExtremeSSD/BioStudy/CodeReview/burstformer/extra/datasets/processed/v1"


binsizes = [500]

with open("extra/results/cell_type_agnostic_bs_bf_para.csv",'w') as w:
    columns = ['bs_auc','bf_auc','fold','eid']
    w.write('\t'.join(columns)+'\n')
    for eid in ["E116","E118","E003"]:
        for fold in [0,1,2,3]:
            print(f"eid:{eid},fold:{fold}")
            # checkpoints = f"checkpoints/{eid}.{fold}.model.pt"
            checkpoints = f"checkpoints/agnostic.{fold}.bs_bf_para.model.pt"
            meta_path = f"extra/datasets/processed/v2/meta_datasets/meta_data_{eid}_delay_1.0_with_cellsize_1.csv"

            seed_everything(seed)
            meta = (
                pd.read_csv(meta_path).sample(frac=1, random_state=seed).reset_index(drop=True)
            )  # load and shuffle.

            # Split genes into two sets (train/val).
            genes = set(meta.gene_id.unique())
            n_genes = len(genes)
            print("Target genes:", len(genes))
            print(f"n_feats_p:{n_feats_p}")

            splits = {
                1: ['chr1', 'chr6', 'chr5', 'chr8', 'chr14', 'chrY'],
                2: ['chr7', 'chr10', 'chr11', 'chr12', 'chr15', 'chr21'],
                3: ['chr2', 'chr3', 'chr4', 'chr16', 'chr18', 'chr20'],
                4: ['chr9', 'chr13', 'chr17', 'chr19', 'chr22', 'chrX'],
            }

            chromosome_splits = {}
            for key, chroms in splits.items():
                for chrom in chroms:
                    chromosome_splits[chrom] = key

            qs = [
                meta[meta.apply(lambda row: chromosome_splits[row['chrom']] == 1,axis=1)].gene_id.tolist(),
                meta[meta.apply(lambda row: chromosome_splits[row['chrom']] == 2,axis=1)].gene_id.tolist(),
                meta[meta.apply(lambda row: chromosome_splits[row['chrom']] == 3,axis=1)].gene_id.tolist(),
                meta[meta.apply(lambda row: chromosome_splits[row['chrom']] == 4,axis=1)].gene_id.tolist(),
            ]

            train_genes = (
                qs[(fold + 0) % 4] + qs[(fold + 1) % 4] + qs[(fold + 2) % 4]
            )
            val_genes = qs[(fold + 3) % 4]



            print(len(train_genes), len(val_genes))

            val_dataset = DeepBurstDataset(
                meta_path,
                npy_dir,
                val_genes,
                i_max,
                binsizes,
                w_prom,
                w_max,
                targets=targets,
                config = config,
                with_gene_id=True
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

            ckpt = torch.load(checkpoints,map_location=DEVICE)
            model.load_state_dict(ckpt["net"])

            # bf Test data evaluation
            # Prepare validation.
            bar = tqdm(enumerate(val_loader, 1), total=len(val_loader))
            gene_ids, val_out, val_label = [], [], []


            # Validation.
            model.eval()
            with torch.no_grad():
                for batch, d in bar:
                    gene_ids += d.pop('gene_id')
                    for k, v in d.items():
                        if isinstance(v, dict):
                            for _k, _v in v.items():
                                v[_k] = _v.to(DEVICE)
                        else:
                            d[k] = v.to(DEVICE)

                    out = model(
                        # d['promoter_seq'],
                        d["promoter_feats"][500],
                        d["promoter_pad_masks"][500],
                    )
                    val_out.append(out.cpu())

                    val_label.append(d["label"].cpu())

            val_out = torch.cat(val_out)
            val_label = torch.cat(val_label)


            val_preds = {}
            for target, pred in zip(targets,torch.chunk(val_out, len(targets), axis=-1)):
                val_preds[target] = pred
            
            val_labels = {}
            for target, label in zip(targets,torch.chunk(val_label, len(targets), axis=-1)):
                val_labels[target] = label   

            # Metrics.

            ckpt = {}
            ckpt['gene_id'] = gene_ids
            description = ""
            line = []
            for target in targets:
                val_score, val_label, val_pred, val_acc, val_auc, val_ap = evaluation(val_preds[target],val_labels[target])

                ckpt[f'{target}_label'] = val_label
                ckpt[f'{target}_score'] = val_score
                ckpt[f'{target}_pred'] = val_pred
                ckpt[f'{target}_val_acc'] = val_acc
                ckpt[f'{target}_val_auc'] = val_auc
                ckpt[f'{target}_val_ap'] = val_ap       
                description += f"{target}: acc={val_acc:.4f}, auc={val_auc:.4f}, ap={val_ap:.4f} " 
                line.append(str(ckpt[f'{target}_val_auc']))

            line += [str(fold),eid]
            w.write('\t'.join(line)+'\n')


if __name__ == '__main__':
    print("Done")