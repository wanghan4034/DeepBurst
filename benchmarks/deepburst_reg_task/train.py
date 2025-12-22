import argparse
import torch
import torch.nn as nn
import os
import pandas as pd
import yaml

from tqdm import tqdm
from scipy import stats
from sklearn import metrics
from typing import List
from benchmarks.DeepBurst_reg_task.data import DeepBurstRegDataset
from benchmarks.DeepBurst_reg_task.net import DeepBurstRegressor
from src.utils.tools import seed_everything
from src.utils.constants import DEVICE
from src.utils.logs import get_logger
from src.model.constants import get_config, MARKS
torch.autograd.set_detect_anomaly(True)

logger = get_logger()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", required=True)
parser.add_argument("-c", "--config", required=True)
parser.add_argument("--exp-id", required=True)
parser.add_argument("-m", "--meta", required=True)
parser.add_argument("-d", "--npy-dir", required=True)
parser.add_argument("--fold", type=int, required=True)
parser.add_argument("--w_prom", type=int, default=40000)
parser.add_argument("--w_max", type=int, default=40000)
parser.add_argument("--binsizes", nargs="+",type=int, default=[500])
parser.add_argument("--remove_marks", nargs="+",type=str, default=[])
parser.add_argument("--keep_marks", nargs="+",type=str, default=[])
parser.add_argument("--add_feature_bin", action="store_true", default=False)
parser.add_argument("--targets", nargs="+",type=str, default=['bs_label','bf_label'])
parser.add_argument("--restore", action="store_true", default=False)

args = parser.parse_args()

#
# Training setup.
#

config = get_config(args.config)

config["exp_id"] = args.exp_id  # Override
config["remove_marks"] = args.remove_marks
config["keep_marks"] = args.keep_marks

if len(config["keep_marks"]) > 0:
    if not len(config["remove_marks"]) > 0:
        remove_marks = [mark for mark in MARKS if mark not in config["keep_marks"] ]
        config["remove_marks"] = remove_marks




seed = config["seed"]
num_epoch = config["num_epoch"]
lr = config["lr"]
bsz = config["bsz"]
gamma = config["gamma"]
patience = int(config['patience'])

i_max = config["i_max"]
w_prom = args.w_prom
w_max = args.w_max

n_feats_p = config['promoter_feats_basic_nums'] - len(config["remove_marks"])
n_feats_pcres = config['pcres_feats_basic_nums'] 
d_emb = config["embed"]["d_model"]
embed_kws = config["embed"]
pairwise_interaction_kws = config["pairwise_interaction"]
regulation_kws = config["regulation"]

d_head = config["d_head"]
targets = args.targets

binsizes = args.binsizes


best_r = 0
best_epoch = 0
early_stop = 1


#
# Setup end.
#

seed_everything(seed)
meta = (
    pd.read_csv(args.meta).sample(frac=1, random_state=seed).reset_index(drop=True)
)  # load and shuffle.


# Split genes into two sets (train/val).
genes = set(meta.gene_id.unique())
n_genes = len(genes)
logger.info(f"Fold:{args.fold}")
logger.info(f"Remove marks:{config['remove_marks']}")
logger.info(f"Target genes:{len(genes)}")
logger.info(f"n_feats_p:{n_feats_p}")

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

meta.drop_duplicates
train_genes = (
    qs[(args.fold + 0) % 4] + qs[(args.fold + 1) % 4] + qs[(args.fold + 2) % 4]
)
val_genes = qs[(args.fold + 3) % 4]



logger.info(f"train:{len(train_genes)}, val:{len(val_genes)}")

train_dataset = DeepBurstRegDataset(
    args.meta,
    args.npy_dir,
    train_genes,
    i_max,
    binsizes,
    w_prom,
    w_max,
    targets=targets,
    config = config,
)
val_dataset = DeepBurstRegDataset(
    args.meta,
    args.npy_dir,
    val_genes,
    i_max,
    binsizes,
    w_prom,
    w_max,
    targets=targets,
    config = config,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=bsz, num_workers=config['num_works'], shuffle=True, drop_last=True
)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz, num_workers=config['num_works'])

model = DeepBurstRegressor(
    n_feats_p,
    d_emb,
    d_head,
    embed_kws=embed_kws,
    binsizes=binsizes,
    seed=42,
    targets=targets,
).to(DEVICE)

if os.path.exists(args.output) and args.restore:
    ckpt = torch.load(args.output)
    model.load_state_dict(ckpt["net"])
    print(f"load pre-trained model {args.output}")

def evaluation(out:'torch.Tensor', label:torch.Tensor):
    score = out.softmax(axis=1)[:, 1]
    pred = out.argmax(axis=1)

    acc = metrics.accuracy_score(label, pred) * 100
    auc = metrics.roc_auc_score(label, score) * 100
    ap = metrics.average_precision_score(label, score) * 100
    return score, label, pred, acc, auc, ap


criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

optimizer.zero_grad()
optimizer.step()

for epoch in range(1, num_epoch):
    # Prepare train.
    bar = tqdm(enumerate(train_loader, 1), total=len(train_loader))
    running_loss = 0.0
    train_out = {target:[] for target in targets}
    train_label = {target:[] for target in targets}    

    # Train.
    model.train()
    for batch, d in bar:
        for k, v in d.items():
            if isinstance(v, dict):
                for _k, _v in v.items():
                    v[_k] = _v.to(DEVICE)
            else:
                d[k] = v.to(DEVICE)

        optimizer.zero_grad()

        out = model(
            d["promoter_feats"][500],
        )
        preds = {}
        for target, pred in zip(targets,torch.chunk(out, len(targets), axis=-1)):
            preds[target] = pred
        
        labels = {}
        for target, label in zip(targets,torch.chunk(d['label'], len(targets), axis=-1)):
            labels[target] = label        
        
        loss = 0
        for target in targets:
            pred = preds[target]
            label = labels[target]
            train_out[target].append(pred.detach().cpu())
            train_label[target].append(label.squeeze().cpu())
            loss += criterion(pred.squeeze(), label.squeeze())

        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().item()
        running_loss += loss


        if batch % 10 == 0:
            batch_loss = running_loss / 10.0
            description = ""
            for target in targets:
                out, labels = map(torch.cat, (train_out[target], train_label[target]))
                batch_r2 = metrics.r2_score(labels, out.squeeze()) * 100
                batch_r = stats.pearsonr(labels, out.squeeze())[0] * 100
                description += f"{target}: batch_r2={batch_r2:.4f}, batch_r={batch_r:.4f} "
            bar.set_description(
            f"E{epoch} {batch_loss:.4f}, lr={get_lr(optimizer)}, {description}"
        )


            running_loss = 0.0
            train_out = {target:[] for target in targets}
            train_label = {target:[] for target in targets}    

    # Prepare validation.
    bar = tqdm(enumerate(val_loader, 1), total=len(val_loader))
    val_out, val_label = [], []

    # Validation.
    model.eval()
    with torch.no_grad():
        for batch, d in bar:
            for k, v in d.items():
                if isinstance(v, dict):
                    for _k, _v in v.items():
                        v[_k] = _v.to(DEVICE)
                else:
                    d[k] = v.to(DEVICE)

            out = model(
                d["promoter_feats"][500],
                # d["promoter_pad_masks"][500],
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

    val_loss = 0
    for target in targets:
        val_pred = val_preds[target]
        val_label = val_labels[target]
        val_loss += criterion(val_pred.squeeze(), val_label.squeeze())

    # Metrics.

    ckpt = {}
    description = ""
    for target in targets:
        batch_r2 = metrics.r2_score(val_labels[target].squeeze(), val_preds[target].squeeze()) * 100
        batch_r = stats.pearsonr(val_labels[target].squeeze(), val_preds[target].squeeze())[0] * 100


        ckpt[f'{target}_label'] = val_label
        ckpt[f'{target}_pred'] = val_pred
        ckpt[f'{target}_val_r2'] = batch_r2
        ckpt[f'{target}_val_r'] = batch_r
        description += f"{target}: r2={batch_r2:.4f}, r={batch_r:.4f} " 

    print(
        f"loss={val_loss:.4f}, lr={get_lr(optimizer)}, {description}"
    )


    if ('bs' in targets):
        if ckpt['bs_val_r']  > best_r:
            ckpt.update({
                "net": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "last_val_loss": val_loss,
            })
            torch.save(ckpt, args.output)
            best_r = ckpt['bs_val_r']
    

    scheduler.step()


torch.cuda.empty_cache()  # 清理显存缓存

