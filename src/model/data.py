import torch
import pandas as pd
import numpy as np
import math
import pickle
from typing import Dict, List, Union
from src.utils.constants import DEVICE
from torch.utils.data import Dataset, DataLoader

MARKS = ["H3K4me1","H3K4me3","H3K9me3","H3K27me3","H3K36me3","H3K27ac","H3K9ac"]

def load_pickle(f):
    with open(f, "rb") as inFile:
        return pickle.load(inFile)


def split_interval_string(interval_string):
    """chrom:start-end -> (chrom, start, end)"""
    chrom, tmp = interval_string.split(":")
    start, end = map(int, tmp.split("-"))

    return chrom, start, end


class BurstPrismaDataset(Dataset):
    def __init__(
        self,
        meta,
        npy_dir,
        gene_ids,
        i_max=8,
        binsizes=[2000, 500, 100],
        w_prom=40000,
        w_max=40000,
        targets = [],
        config = None,
        with_gene_id = False
    ):
        super(BurstPrismaDataset, self).__init__()

        self.npy_dir = npy_dir
        self.gene_ids = gene_ids  # List of ENSGs.

        self.meta = pd.read_csv(meta)
        self.config = config
        self.with_gene_id = with_gene_id


        self.ensg2label = {r['gene_id']: [r[target] for target in targets] for r in self.meta.to_records()}

        self.ensg2tss, self.ensg2pcres, self.ensg2scores = {}, {}, {}
        for r in self.meta.to_records():
            self.ensg2tss[r['gene_id']] = (r['chrom'], r['start'], r['end'], r['strand'], r['eid'])

            if not pd.isnull(r['neighbors']):
                self.ensg2pcres[r['gene_id']] = r['neighbors'].split(";")
                self.ensg2scores[r['gene_id']] = [float(s) for s in r['scores'].split(";")]
            else:
                self.ensg2pcres[r['gene_id']] = []
                self.ensg2scores[r['gene_id']] = []

        self.i_max = i_max  # Maximum number of cis-interacting pCREs.
        self.binsizes = binsizes  # List of genomic bin sizes to use.
        self.w_prom = w_prom  # Promoter window size.
        self.w_max = w_max  # Maximum size of pCRE to consider.

    def _bin_and_pad(self, x, bin_size, max_n_bins, with_sequence = True):
        """Given a 2D tensor x, make binned tensor by
        taking average values of `bin_size` consecutive values.
        Appropriately pad by
        left_pad = ceil((max_n_bins - n_bins) / 2)
        right_pad = floor((max_n_bins - n_bins) / 2)
        """
        l = x.size(1)
        n_bins = min(math.ceil(l / bin_size), max_n_bins)

        mark_idxes = [idx for idx, mark in enumerate(MARKS) if mark not in self.config['remove_marks']]
        perturbation_marks_idxes = [idx for idx, mark in enumerate(MARKS) if mark in self.config['masked_marks']]
        

        # Binning.
        x_binned = []

        for i in range(n_bins):
            if i in self.config['marked_bin_idxes']:
                x[perturbation_marks_idxes, i * bin_size : (i + 1) * bin_size] = x[perturbation_marks_idxes, i * bin_size : (i + 1) * bin_size] * self.config['perturbation_strength']

            b = torch.mean(x[mark_idxes, i * bin_size : (i + 1) * bin_size], axis=-1, keepdims=True)
            b = torch.log1p(b)
            x_binned.append(b)

        x_binned = torch.cat(x_binned, axis=-1)

        # Padding.
        left_pad = math.ceil((max_n_bins - n_bins) / 2)
        right_pad = math.floor((max_n_bins - n_bins) / 2)

        x_binned = torch.cat(
            [
                torch.zeros([x_binned.size(0), left_pad]),
                x_binned,
                torch.zeros([x_binned.size(0), right_pad]),
            ],
            dim=1,
        )

        return x_binned, left_pad, n_bins, right_pad

    def _get_region_representation(
        self, eid, chrom, start, end, bin_size, max_n_bins, strand="+", window=None
    ):
        x = torch.tensor(np.load(f"{self.npy_dir}/{eid}/regions/{chrom}:{start}-{end}.npy")).float()

        if window is not None:
            x = x[:, 20000 - window // 2 : 20000 + window // 2]
        x, left_pad, n_bins, right_pad = self._bin_and_pad(x, bin_size, max_n_bins, with_sequence=int(self.config['promoter_with_sequence']))

        if strand == "+":
            return x, left_pad, n_bins, right_pad
        else:
            return torch.fliplr(x), right_pad, n_bins, left_pad


    def _get_promoter_representation(
        self, eid, gene_id, bin_size, max_n_bins, strand="+", window=None
    ):
        x = torch.tensor(np.load(f"{self.npy_dir}/{eid}/regions/{gene_id}.npy")).float()

        if window is not None:
            x = x[:, 20000 - window // 2 : 20000 + window // 2]
        x_seq = x[-self.config['base_nums']:,:]
        x_binned, left_pad, n_bins, right_pad = self._bin_and_pad(x, bin_size, max_n_bins,with_sequence=int(self.config['pcres_with_sequence']))

        if strand == "+":
            return x_seq, x_binned, left_pad, n_bins, right_pad
        else:
            return torch.fliplr(x_seq),torch.fliplr(x_binned), right_pad, n_bins, left_pad


    def __getitem__(self, i):
        gene_id = self.gene_ids[i]
        item = self.get_sample(gene_id)
        if self.with_gene_id:
            item['gene_id'] = gene_id

        return item

    def __len__(self):
        return len(self.gene_ids)

    def get_sample(self, gene_id, add_bsz_dim = False):

        if gene_id not in self.ensg2tss:
            print(gene_id)

        item = {}

        item["label"] = torch.tensor(self.ensg2label[gene_id]).long()

        item["promoter_feats"] = {}
        item["promoter_pad_masks"] = {}
        item["pcre_feats"] = {}
        item["pcre_pad_masks"] = {}
        item["interaction_masks"] = {}

        chrom_p, start_p, end_p, strand_p, eid = self.ensg2tss[gene_id]
        # start_p, end_p = start_p - 20000, start_p + 20000

        for binsize in self.binsizes:
            max_n_bins = self.w_max // binsize

            x_seq, x_p, left_pad_p, n_bins_p, right_pad_p = self._get_promoter_representation(
                eid,
                gene_id,
                binsize,
                max_n_bins,
                strand_p,
                window=self.w_prom,
            )

            x_p = x_p.permute(1, 0).unsqueeze(0)  # 1 x max_n_bins x n_feats

            mask_p = torch.ones([1, max_n_bins, max_n_bins], dtype=torch.bool)
            mask_p[
                0,
                left_pad_p : left_pad_p + n_bins_p,
                left_pad_p : left_pad_p + n_bins_p,
            ] = 0
            mask_p.unsqueeze_(0)

            item["promoter_seq"] = x_seq
            item["promoter_feats"][binsize] = x_p
            item["promoter_pad_masks"][binsize] = mask_p


        if add_bsz_dim:
            item = get_bsz_dataset(item)

        return item

def get_bsz_dataset(samples,bsz=None):
    if isinstance(samples,Dict):
        bsz = 1
    if not bsz:
        bsz = len(samples)

    sample_dataset = SampleDataset(samples)
    sample_loader = DataLoader(sample_dataset, batch_size=bsz)
    item = list(sample_loader)[0]
    for k, v in item.items():
        if isinstance(v, dict):
            for _k, _v in v.items():
                v[_k] = _v.to(DEVICE)
        else:
            item[k] = v.to(DEVICE)
    return item

class SampleDataset(Dataset):
    def __init__(self,samples: Union[List[Dict],Dict]) -> None:
        super(SampleDataset, self).__init__()
        if isinstance(samples,Dict):
            samples = [samples]

        self.samples = samples

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    import tqdm

    meta = "extra/datasets/meta_datasets/burst_bf_meta_data_E118.csv"
    gene_ids = pd.read_csv(meta)['gene_id'].unique()
    npy_dir = "extra/datasets/processed/E118"

    dataset = BurstPrismaDataset(meta, npy_dir, gene_ids,n_feats=11)
    loader = DataLoader(dataset, batch_size=8, num_workers=1, shuffle=False)

    for i, d in enumerate(loader):
        for binsize in [2000, 500, 100]:
            print(f'{d["promoter_feats"][binsize].shape=}')
            print(f'{d["promoter_pad_masks"][binsize].shape=}')
            print(f'{d["pcre_feats"][binsize].shape=}')
            print(f'{d["pcre_pad_masks"][binsize].shape=}')
            print(f'{d["interaction_masks"][binsize].shape=}')

        print(f'{d["interaction_freq"].shape=}')

        break
