import os
import torch
import pandas as pd
import numpy as np
import math
import pickle
from typing import Dict, List, Union, Tuple, Optional
from src.utils.constants import DEVICE
from torch.utils.data import Dataset, DataLoader
from src.model.constants import MARKS, PERTURBATION_STRENGTH, PERTURBATION_REGION


def load_pickle(f: str):
    with open(f, "rb") as inFile:
        return pickle.load(inFile)


def split_interval_string(interval_string: str) -> Tuple[str, int, int]:
    """'chrom:start-end' -> (chrom, start, end)"""
    chrom, tmp = interval_string.split(":")
    start, end = map(int, tmp.split("-"))
    return chrom, start, end


# ------------------------- Core helpers ------------------------- #

def _safe_config(config: Optional[Dict]) -> Dict:
    """Provide safe defaults for keys used below."""
    if config is None:
        return {"remove_marks": [], "masked_marks": {}}
    if "remove_marks" not in config or config["remove_marks"] is None:
        config["remove_marks"] = []
    if "masked_marks" not in config or config["masked_marks"] is None:
        config["masked_marks"] = {}
    return config


def _build_mask_mat(
    n_used: int,
    n_bins: int,
    marks_in_use: List[str],
    masked_cfg: Dict
) -> torch.Tensor:
    """
    构建 (n_used, n_bins) 的乘性掩码矩阵。默认 1.0；对每个 mark/区域设置 strength。
    - 支持 region="all" 或 [bin_idx,...]
    - 会自动忽略越界的 bin
    """
    mask_mat = torch.ones((n_used, n_bins), dtype=torch.float32)
    if not isinstance(masked_cfg, dict) or n_bins == 0 or n_used == 0:
        return mask_mat

    row_of = {m: i for i, m in enumerate(marks_in_use)}
    for mark_name, spec in masked_cfg.items():
        row = row_of.get(mark_name, None)
        if row is None:
            continue
        strength = float(spec.get(PERTURBATION_STRENGTH, 1.0))
        regions = spec.get(PERTURBATION_REGION, [])
        if regions == "all":
            mask_mat[row, :] = strength
        elif isinstance(regions, (list, tuple)):
            valid = [int(b) for b in regions if 0 <= int(b) < n_bins]
            if valid:
                idx = torch.tensor(valid, dtype=torch.long)
                mask_mat[row, idx] = strength
    return mask_mat


def _bin_log_pad_with_mask(
    x_used: torch.Tensor,          # (n_used, L_bp)
    bin_size: int,
    max_n_bins: int,
    marks_in_use: List[str],
    masked_cfg: Dict
) -> Tuple[torch.Tensor, int, int, int]:
    """
    向量化：先均值分箱，再按 mark×bin 乘掩码，最后 log1p + 居中 padding。
    返回：(n_used, max_n_bins), left_pad, n_bins, right_pad
    """
    L = x_used.size(1)
    n_bins = min(int(math.ceil(L / bin_size)), max_n_bins)
    eff_len = n_bins * bin_size

    if eff_len == 0:
        left_pad = int(math.ceil(max_n_bins / 2))
        right_pad = int(math.floor(max_n_bins / 2))
        return torch.zeros((x_used.size(0), max_n_bins), dtype=torch.float32), left_pad, 0, right_pad

    # (n_used, n_bins, bin_size) -> mean over last
    x_eff = x_used[:, :eff_len]
    x_pool = x_eff.reshape(x_used.size(0), n_bins, bin_size).mean(dim=-1)  # (n_used, n_bins)

    # 掩码（mark×bin）
    mask_mat = _build_mask_mat(x_used.size(0), n_bins, marks_in_use, masked_cfg)  # (n_used, n_bins)
    if mask_mat.device != x_pool.device:
        mask_mat = mask_mat.to(x_pool.device)
    x_logged = torch.log1p(x_pool * mask_mat)

    # 居中 padding
    left_pad = int(math.ceil((max_n_bins - n_bins) / 2))
    right_pad = int(math.floor((max_n_bins - n_bins) / 2))
    if left_pad > 0 or right_pad > 0:
        x_logged = torch.cat(
            [
                torch.zeros((x_logged.size(0), left_pad), dtype=torch.float32, device=x_logged.device),
                x_logged,
                torch.zeros((x_logged.size(0), right_pad), dtype=torch.float32, device=x_logged.device),
            ],
            dim=1,
        )  # (n_used, max_n_bins)

    return x_logged, left_pad, n_bins, right_pad


# ------------------------- Datasets ------------------------- #

class DeepBurstDataset(Dataset):
    """
    Build multi-scale promoter (and optional pCRE) features.
    For each binsize:
      1) optional perturbation masks (per mark × bin strength);
      2) avg-pool binning and center-pad to fixed length;
      3) 2D padding masks for attention blocks.

    Expected .npy layout:
      - promoter:  {npy_dir}/{eid}/regions/{gene_id}.npy   -> (n_marks(+bases), L_bp)
      - region:    {npy_dir}/{eid}/regions/{chrom}:{start}-{end}.npy
    """
    def __init__(
        self,
        meta: str,
        npy_dir: str,
        gene_ids: List[str],
        i_max: int = 8,
        binsizes: List[int] = [2000, 500, 100],
        w_prom: int = 40000,
        w_max: int = 40000,
        targets: List[str] = [],
        config: Optional[Dict] = None,
        with_gene_id: bool = False,
    ):
        super().__init__()
        self.npy_dir = npy_dir
        self.gene_ids = list(gene_ids)

        self.meta = pd.read_csv(meta)
        self.config = _safe_config(config)
        self.with_gene_id = with_gene_id

        # labels
        self.ensg2label = (
            {r["gene_id"]: [r[t] for t in targets] for r in self.meta.to_records()}
            if targets else {}
        )

        # gene_id -> (chrom, start, end, strand, eid)
        self.ensg2tss: Dict[str, Tuple[str, int, int, str, str]] = {}
        for r in self.meta.to_records():
            self.ensg2tss[r["gene_id"]] = (r["chrom"], r["start"], r["end"], r["strand"], r["eid"])

        self.i_max = i_max
        self.binsizes = list(binsizes)
        self.w_prom = w_prom
        self.w_max = w_max

    # ================= vectorized bin+mask ================= #

    def _bin_and_pad(
        self, x: torch.Tensor, bin_size: int, max_n_bins: int
    ) -> Tuple[torch.Tensor, int, int, int]:
        """
        向量化 bin+mask+log+pad（PyTorch版）。
        x shape 期望为 (n_feats, L_bp)，其中前 len(MARKS) 行为 histone marks。
        """
        assert x.ndim == 2, f"x must be 2D (n_feats, L_bp), got {x.shape}"
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.to(torch.float32)

        remove_marks = set(self.config.get("remove_marks", []))
        mark_idxes = [i for i, m in enumerate(MARKS) if m not in remove_marks]
        marks_in_use = [MARKS[i] for i in mark_idxes]

        # 只对使用的 marks 做 pooling+mask；序列等其它通道在外层函数单独返回
        x_used = x[mark_idxes, :]  # (n_used, L_bp)

        x_binned, left_pad, n_bins, right_pad = _bin_log_pad_with_mask(
            x_used, bin_size, max_n_bins, marks_in_use, self.config.get("masked_marks", {})
        )
        return x_binned, left_pad, n_bins, right_pad

    # ================ region / promoter views ================= #

    def _get_region_representation(
        self, eid: str, chrom: str, start: int, end: int,
        bin_size: int, max_n_bins: int, strand: str = "+", window: Optional[int] = None
    ) -> Tuple[torch.Tensor, int, int, int]:
        x_np = np.load(f"{self.npy_dir}/{eid}/regions/{chrom}:{start}-{end}.npy")
        x = torch.from_numpy(x_np).to(torch.float32)  # (n_feats, L_bp)
        if window is not None:
            x = x[:, 20000 - window // 2 : 20000 + window // 2]
        x_binned, left_pad, n_bins, right_pad = self._bin_and_pad(x, bin_size, max_n_bins)

        if strand == "+":
            return x_binned, left_pad, n_bins, right_pad
        else:
            return torch.fliplr(x_binned), right_pad, n_bins, left_pad

    def _get_promoter_representation(
        self, eid: str, gene_id: str, bin_size: int, max_n_bins: int,
        strand: str = "+", window: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
        x_np = np.load(f"{self.npy_dir}/{eid}/regions/{gene_id}.npy")
        x = torch.from_numpy(x_np).to(torch.float32)  # (n_feats, L_bp)

        if window is not None:
            x = x[:, 20000 - window // 2 : 20000 + window // 2]

        # 序列通道（保留原有接口：最后 base_nums 行）
        base_nums = int(self.config.get("base_nums", 0))
        x_seq = x[-base_nums:, :] if base_nums > 0 else torch.empty((0, x.size(1)), dtype=torch.float32)

        x_binned, left_pad, n_bins, right_pad = self._bin_and_pad(x, bin_size, max_n_bins)

        if strand == "+":
            return x_seq, x_binned, left_pad, n_bins, right_pad
        else:
            return torch.fliplr(x_seq), torch.fliplr(x_binned), right_pad, n_bins, left_pad

    # ================= standard Dataset API ================= #

    def __getitem__(self, i: int) -> Dict:
        gene_id = self.gene_ids[i]
        item = self.get_sample(gene_id)
        if self.with_gene_id:
            item["gene_id"] = gene_id
        return item

    def __len__(self) -> int:
        return len(self.gene_ids)

    def get_sample(self, gene_id: str, add_bsz_dim: bool = False) -> Dict:
        if gene_id not in self.ensg2tss:
            # 未在 meta 中找到该 gene，返回空样本或抛错均可；此处保持打印提醒
            print(f"[Warn] gene_id not in meta: {gene_id}")

        item: Dict[str, Union[torch.Tensor, Dict]] = {}
        if self.ensg2label:
            item["label"] = torch.tensor(self.ensg2label[gene_id]).long()

        item["promoter_feats"] = {}
        item["promoter_pad_masks"] = {}

        chrom_p, start_p, end_p, strand_p, eid = self.ensg2tss[gene_id]

        for binsize in self.binsizes:
            max_n_bins = self.w_max // binsize
            x_seq, x_p, left_pad_p, n_bins_p, right_pad_p = self._get_promoter_representation(
                eid, gene_id, binsize, max_n_bins, strand_p, window=self.w_prom
            )

            # (1, max_n_bins, n_feats)  —— 注意：x_p 维度为 (n_used_marks, max_n_bins)
            x_p = x_p.permute(1, 0).unsqueeze(0)

            # 2D pad mask (1,1,max_n_bins,max_n_bins)，有效区域为 0
            mask_p = torch.ones([1, max_n_bins, max_n_bins], dtype=torch.bool)
            mask_p[0,
                   left_pad_p : left_pad_p + n_bins_p,
                   left_pad_p : left_pad_p + n_bins_p] = 0
            mask_p = mask_p.unsqueeze(0)

            item["promoter_feats"][binsize] = x_p
            item["promoter_pad_masks"][binsize] = mask_p

        if add_bsz_dim:
            item = get_bsz_dataset(item)

        return item


# -------- batching helper -------- #

def get_bsz_dataset(samples: Union[List[Dict], Dict], bsz: Optional[int] = None) -> Dict:
    if isinstance(samples, Dict):
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
    def __init__(self, samples: Union[List[Dict], Dict]) -> None:
        super().__init__()
        self.samples = samples if isinstance(samples, list) else [samples]

    def __getitem__(self, index: int) -> Dict:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


# ------------------------- Optional: EnformerDataset ------------------------- #

class EnformerDataset(Dataset):
    """
    Dataset for promoter-centered histone modification signals saved as .npy files.
    Each .npy file has shape (40000, C).
    """

    def __init__(
        self,
        meta_path: str,
        gene_ids: List[str],
        npy_dir: str,
        targets: Optional[List[str]] = None,
        binsizes: List[int] = [2000, 500, 100],
        w_prom: int = 40000,
    ):
        self.meta = pd.read_csv(meta_path)
        self.gene_ids = list(gene_ids)
        self.npy_dir = npy_dir
        self.binsizes = list(binsizes)
        self.w_prom = w_prom
        self.ensg2label = (
            {r["gene_id"]: [r[t] for t in targets] for r in self.meta.to_records()}
            if targets else {}
        )
        self.targets = targets

    def __len__(self) -> int:
        return len(self.gene_ids)

    def _bin_and_pad(self, x: np.ndarray, bin_size: int, max_n_bins: int) -> Tuple[torch.Tensor, int, int, int]:
        """
        NumPy 向量化：avg-pool -> log1p -> 居中 pad
        x: (L_bp, C)
        """
        L, C = x.shape
        n_bins = min(int(np.ceil(L / bin_size)), max_n_bins)
        eff_len = n_bins * bin_size

        if eff_len == 0:
            left_pad = int(np.ceil(max_n_bins / 2))
            right_pad = int(np.floor(max_n_bins / 2))
            return torch.zeros((1, max_n_bins, C), dtype=torch.float32).squeeze(0), left_pad, 0, right_pad

        x_eff = x[:eff_len, :]
        x_pool = x_eff.reshape(n_bins, bin_size, C).mean(axis=1)  # (n_bins, C)
        x_logged = np.log1p(x_pool)

        left_pad = int(np.ceil((max_n_bins - n_bins) / 2))
        right_pad = int(np.floor((max_n_bins - n_bins) / 2))
        x_padded = np.pad(
            x_logged,
            ((left_pad, right_pad), (0, 0)),
            mode="constant",
            constant_values=0.0,
        ).astype(np.float32)

        return torch.from_numpy(x_padded), left_pad, n_bins, right_pad

    def __getitem__(self, idx: int) -> Dict:
        gene_id = self.gene_ids[idx]
        npy_path = os.path.join(self.npy_dir, gene_id + ".npy")
        x = np.load(npy_path)  # (L_bp, C)

        item: Dict = {"gene_id": gene_id, "promoter_feats": {}, "promoter_pad_masks": {}}
        if self.targets:
            item["label"] = torch.tensor(self.ensg2label[gene_id]).long()

        for binsize in self.binsizes:
            max_n_bins = self.w_prom // binsize
            x_binned, left_pad, n_bins, right_pad = self._bin_and_pad(x, binsize, max_n_bins)
            x_binned = x_binned.unsqueeze(0)  # (1, max_n_bins, C)

            mask_p = torch.ones([1, max_n_bins, max_n_bins], dtype=torch.bool)
            mask_p[0,
                   left_pad : left_pad + n_bins,
                   left_pad : left_pad + n_bins] = 0
            mask_p = mask_p.unsqueeze(0)  # (1,1,max_n_bins,max_n_bins)

            item["promoter_feats"][binsize] = x_binned
            item["promoter_pad_masks"][binsize] = mask_p

        return item