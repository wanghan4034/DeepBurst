import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from intervaltree import IntervalTree
from typing import List
from src.data.features.genomics import get_cres_infos, get_gene_infos
from src.model.data import BurstFormerDataset
from src.model.net import BurstFormer
from src.utils.constants import DEVICE
from src.utils.tools import seed_everything
from src.model.constants import get_config


torch.autograd.set_detect_anomaly(True)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

#
# Training setup.
#
config_path = "configs/default.yaml"
config = get_config(config_path)
config['masked_marks'] = []
# add_feature_bin = True
add_feature_bin = False

eid = 'E116'
remove_marks = None

if remove_marks:
    config["remove_marks"] = [remove_marks]
else:
    config["remove_marks"] = []


seed = config["seed"]

bsz = config["bsz"]
gamma = config["gamma"]

i_max = config["i_max"]
w_prom = config["w_prom"]
w_max = config["w_max"]

n_feats_p = config['promoter_feats_basic_nums'] - len(config["remove_marks"])
n_feats_pcres = config['pcres_feats_basic_nums'] 
d_emb = config["embed"]["d_model"]
embed_kws = config["embed"]
pairwise_interaction_kws = config["pairwise_interaction"]
regulation_kws = config["regulation"]
d_head = config["d_head"]
targets = ['bs_label','bf_label']
# npy_dir = "extra/datasets/processed/v1"
npy_dir = f"/Volumes/ExtremeSSD/BioStudy/CodeReview/burstformer/extra/datasets/processed/v1"

models = {}
datasets = {}
for fold in [0,1,2,3]:

    binsizes = [500]
    checkpoints = f"checkpoints/{eid}.{fold}.bs_bf_para.model.pt"


    meta_path = f"extra/datasets/processed/v1/meta_datasets/meta_data_{eid}.csv"

    #
    # Setup end.
    #

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

    val_dataset = BurstFormerDataset(
        meta_path,
        npy_dir,
        val_genes,
        i_max,
        binsizes,
        w_prom,
        w_max,
        config = config,
        with_gene_id=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz, num_workers=config['num_works'])

    model = BurstFormer(
        n_feats_p,
        d_emb,
        d_head,        embed_kws=embed_kws,
        binsizes=binsizes,
        seed=42,
        targets=targets,
    ).to(DEVICE)

    ckpt = torch.load(checkpoints,map_location=DEVICE)
    model.load_state_dict(ckpt["net"])
    models[fold] = model 
    datasets[fold] = val_dataset



# ---------------------- 构建 IntervalTree ----------------------
def build_cres_trees(cres_infos):
    """
    将 cCRE 区域按照染色体组织为 IntervalTree，加速 overlap 查询
    """
    trees = {}
    for cres in cres_infos.values():
        chrom = cres.region.chrom
        start = int(cres.region.start.location)
        end = int(cres.region.end.location)
        if chrom not in trees:
            trees[chrom] = IntervalTree()
        trees[chrom].addi(start, end, cres)
    return trees


# ---------------------- 提取 Attention 权重 ----------------------
def get_attention_weights(gene_id, dataset: BurstFormerDataset, model: BurstFormer, binsize=500, dims=None):
    sample = dataset.get_sample(gene_id, add_bsz_dim=True)
    x = sample["promoter_feats"][binsize]
    mask = sample["promoter_pad_masks"][binsize]
    x, mask = model.generate_transformer_inputs(x, mask, binsize)
    attention_weights = np.squeeze(model.get_attention_weights(binsize, x, mask))
    if isinstance(dims, int):
        attention_weights = attention_weights[dims]
    else:
        attention_weights = np.sum(attention_weights, axis=0)
    return attention_weights


# ---------------------- 提取高注意力区域 ----------------------
def extract_attention_regions(weights, chrom, center, strand, threshold=0.05, bin_size=500) -> List[tuple]:
    if strand == '-':
        weights = np.flip(weights)

    def _dist_to_tss(idx):
        return bin_size * (idx - 40)  # 第40个bin代表TSS位置

    regions, region = [], []
    for i, w in enumerate(weights):
        if w > threshold:
            region.append(i)
        elif region:
            regions.append(region)
            region = []
    if region:
        regions.append(region)

    genomic_regions = []
    for region in regions:
        start = center + _dist_to_tss(region[0])
        end = center + _dist_to_tss(region[-1] + 1)
        genomic_regions.append((chrom, start, end))
    return genomic_regions


# ---------------------- 主分析逻辑 ----------------------
def compute_attention_overlap(data, datasets, models, cres_infos, gene_infos,threshold=0.05, out_path=None, promoter_window=20000):
    cres_trees = build_cres_trees(cres_infos)
    records = []
    dims = [0, 1]
    # data = data[data['gene_id']=='ENSG00000163320'] ## 测试代码

    for record in tqdm(data.to_records(), desc=f"Computing overlaps (threshold={threshold})"):
        gene_id = record['gene_id']
        gene_info = gene_infos[gene_id]
        chrom = gene_info.chrom
        strand = gene_info.strand
        center = gene_info.transcript_start_site.location
        fold = record['fold']

        for dim in dims:
            weights = get_attention_weights(gene_id, datasets[fold], models[fold], dims=dim)
            if strand == '-':
                weights = np.flip(weights, axis=1)
            promoter_att = weights[40, :]  # 关注 promoter attention 轨迹
            high_att_regions = extract_attention_regions(promoter_att, chrom, center, strand, threshold)

            # === 1️  Attention 区域 ===
            att_regions = high_att_regions
            att_intervals = [(s, e) for _, s, e in att_regions]

            # === 2️  非 attention 区域（在 ±20kb 内，去除 attention 区间）===
            full_range = [(center - promoter_window, center + promoter_window)]
            non_att_intervals = []
            for s_full, e_full in full_range:
                current = s_full
                for s_att, e_att in sorted(att_intervals):
                    if current < s_att:
                        non_att_intervals.append((current, s_att))
                    current = max(current, e_att)
                if current < e_full:
                    non_att_intervals.append((current, e_full))

            # Helper for overlap 计算
            def _overlap_regions(regions, region_type):
                overlaps = []
                for start, end in regions:
                    if chrom not in cres_trees:
                        continue
                    hits = cres_trees[chrom].overlap(start, end)
                    for h in hits:
                        cres = h.data
                        overlaps.append({
                            'gene_id': gene_id,
                            'chrom': chrom,
                            'region_start': start,
                            'region_end': end,
                            'region_type': region_type,
                            'dim': dim,
                            'cres_id': cres.id,
                            'cres_type': cres.type,
                            'cres_start': cres.region.start.location,
                            'cres_end': cres.region.end.location,
                            'overlap_start': max(start, cres.region.start.location),
                            'overlap_end': min(end, cres.region.end.location),
                            'overlap_length': min(end, cres.region.end.location) - max(start, cres.region.start.location)
                        })
                return overlaps

            att_overlaps = _overlap_regions(att_intervals, "attention")
            non_overlaps = _overlap_regions(non_att_intervals, "non_attention")

            if att_overlaps:
                records.extend(att_overlaps)
            if non_overlaps:
                records.extend(non_overlaps)
            # 即使都无 overlap，也保留空行（方便统计）
            if not (att_overlaps or non_overlaps):
                records.append({
                    'gene_id': gene_id,
                    'chrom': chrom,
                    'region_type': 'none',
                    'region_start': None,
                    'region_end': None,
                    'cres_id': None,
                    'cres_type': None,
                    'overlap_length': 0
                })

    df_out = pd.DataFrame(records)
    if out_path:
        df_out.to_csv(out_path, index=False)
        print(f"[Saved] Overlap table → {out_path}")
    return df_out

# ---------------------- 汇总统计 ----------------------
def summarize_overlap(df):
    summary_gene = df.groupby('gene_id')['cres_id'].count().reset_index(name='n_overlaps')
    summary_type = df['cres_type'].value_counts(normalize=True).rename_axis('cres_type').reset_index(name='percent')

    print(f"Total genes: {len(summary_gene)}, total overlaps: {len(df)}")
    print("Top cCRE types:")
    print(summary_type.head(10))

    return summary_gene, summary_type


# ---------------------- 示例运行 ----------------------
if __name__ == "__main__":
    # eid = "E003"
    data = pd.read_csv(f"extra/datasets/results/predictions_bs_bf.csv")
    data = data[data['eid'] == eid]

    cres_infos = get_cres_infos('extra/datasets/genomic/GRCh19-cCREs.bed')
    gene_infos = get_gene_infos('extra/datasets/genomic/genes.bed')

    df_overlap = compute_attention_overlap(data, datasets, models, cres_infos, gene_infos,
                                           threshold=0.1,
                                           out_path=f"extra/datasets/results/attention_overlap_{eid}.csv")

    summarize_overlap(df_overlap)