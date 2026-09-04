import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig


class HistoneExtractor:
    def __init__(
        self,
        ensg2tss_path,
        bigwig_dir,
        eid="E003",
        marks=None,
        window=20000,
        bin_size=500,
        seed=20240531,
        log1p=True,
    ):
        self.ensg2tss_path = Path(ensg2tss_path)
        self.bigwig_dir = Path(bigwig_dir)
        self.eid = eid
        self.window = int(window)
        self.bin_size = int(bin_size)
        self.seed = int(seed)
        self.log1p = bool(log1p)

        self.marks = marks or [
            "H3K4me1",
            "H3K4me3",
            "H3K9me3",
            "H3K27me3",
            "H3K36me3",
            "H3K27ac",
            "H3K9ac",
        ]

        self.n_bins = 2 * self.window // self.bin_size

        if 2 * self.window % self.bin_size != 0:
            raise ValueError("2 * window must be divisible by bin_size")

        with open(self.ensg2tss_path, "rb") as f:
            self.ensg2tss = pickle.load(f)

        self.rng = random.Random(self.seed)

        self.bws = {}
        for mark in self.marks:
            bw_path = self.bigwig_dir / f"{self.eid}-{mark}.bw"
            if not bw_path.exists():
                raise FileNotFoundError(f"Missing BigWig file: {bw_path}")
            self.bws[mark] = pyBigWig.open(str(bw_path))

    @staticmethod
    def dedup_tss(tss_list):
        """
        Deduplicate TSS tuples while preserving original order.

        Expected tuple format:
            (chrom, start, end, strand)
        """
        seen = set()
        unique = []

        for chrom, start, end, strand in tss_list:
            item = (str(chrom), int(start), int(end), str(strand))

            if item in seen:
                continue

            seen.add(item)
            unique.append(item)

        return unique

    @staticmethod
    def _normalize_tss_list(tss_list):
        """
        Normalize raw TSS tuples.

        Expected tuple format:
            (chrom, start, end, strand)
        """
        normalized = []

        for chrom, start, end, strand in tss_list:
            normalized.append(
                (
                    str(chrom),
                    int(start),
                    int(end),
                    str(strand),
                )
            )

        return normalized

    def get_all_tss(
        self,
        gene_id,
        unique=True,
        as_dataframe=True,
    ):
        """
        Interface 1:
        Input one gene ID and return all candidate TSSs for this gene.

        Parameters
        ----------
        gene_id:
            ENSG gene ID.

        unique:
            If True, return unique TSSs after deduplication.
            If False, return raw transcript-level TSS records.

        as_dataframe:
            If True, return a pandas DataFrame.
            If False, return a list of dictionaries.

        Returns
        -------
        tss_df or records:
            Columns include:
                gene_id
                tss_index
                chrom
                start
                end
                strand
                tss
                is_first_tss
                distance_to_first_tss_bp
                n_tss_raw
                n_tss_unique
        """
        if gene_id not in self.ensg2tss:
            raise KeyError(f"{gene_id} not found in ensg2tss")

        raw_tss_list = self._normalize_tss_list(self.ensg2tss[gene_id])
        unique_tss_list = self.dedup_tss(raw_tss_list)

        if len(unique_tss_list) == 0:
            raise ValueError(f"{gene_id} has no valid TSS")

        if unique:
            tss_list = unique_tss_list
        else:
            tss_list = raw_tss_list

        first_chrom, first_start, first_end, first_strand = unique_tss_list[0]
        first_tss = int(first_start)

        records = []

        for i, (chrom, start, end, strand) in enumerate(tss_list):
            tss = int(start)

            if chrom == first_chrom and strand == first_strand:
                distance_to_first_tss_bp = abs(tss - first_tss)
            else:
                distance_to_first_tss_bp = np.nan

            records.append(
                {
                    "gene_id": gene_id,
                    "tss_index": i,
                    "chrom": chrom,
                    "start": int(start),
                    "end": int(end),
                    "strand": strand,
                    "tss": int(tss),
                    "is_first_tss": i == 0,
                    "first_chrom": first_chrom,
                    "first_start": int(first_start),
                    "first_end": int(first_end),
                    "first_strand": first_strand,
                    "first_tss": int(first_tss),
                    "distance_to_first_tss_bp": distance_to_first_tss_bp,
                    "n_tss_raw": len(raw_tss_list),
                    "n_tss_unique": len(unique_tss_list),
                    "is_unique_output": bool(unique),
                }
            )

        if as_dataframe:
            return pd.DataFrame(records)

        return records

    def _parse_tss_input(
        self,
        tss=None,
        chrom=None,
        start=None,
        end=None,
        strand=None,
        gene_id=None,
        tss_index=None,
    ):
        """
        Parse flexible TSS input.

        Supported formats
        -----------------
        1. dict or pandas Series:
            {
                "chrom": "chr1",
                "start": 12345,
                "end": 12346,
                "strand": "+"
            }

            Also supports "tss" or "selected_tss" instead of "start".

        2. tuple/list with 4 elements:
            ("chr1", 12345, 12346, "+")

        3. tuple/list with 3 elements:
            ("chr1", 12345, "+")

        4. explicit arguments:
            chrom="chr1", start=12345, end=12346, strand="+"
        """
        if tss is not None:
            if isinstance(tss, pd.Series):
                tss = tss.to_dict()

            if isinstance(tss, dict):
                gene_id = tss.get("gene_id", gene_id)
                tss_index = tss.get("tss_index", tss_index)

                chrom = tss.get("chrom", chrom)

                if "start" in tss:
                    start = tss["start"]
                elif "tss" in tss:
                    start = tss["tss"]
                elif "selected_tss" in tss:
                    start = tss["selected_tss"]
                elif start is None:
                    raise ValueError(
                        "TSS dict must contain one of: 'start', 'tss', or 'selected_tss'"
                    )

                end = tss.get("end", tss.get("tss_end", end))
                strand = tss.get("strand", strand)

            elif isinstance(tss, (tuple, list)):
                if len(tss) == 4:
                    chrom, start, end, strand = tss
                elif len(tss) == 3:
                    chrom, start, strand = tss
                    end = int(start) + 1
                else:
                    raise ValueError(
                        "Tuple/list TSS input must have length 3 or 4: "
                        "(chrom, tss, strand) or (chrom, start, end, strand)"
                    )

            else:
                raise TypeError(
                    "tss must be a dict, pandas Series, tuple, list, or None"
                )

        if chrom is None:
            raise ValueError("chrom is required")

        if start is None:
            raise ValueError("start or tss position is required")

        chrom = str(chrom)
        start = int(start)

        if end is None:
            end = start + 1
        else:
            end = int(end)

        if strand is None:
            strand = "+"

        strand = str(strand)

        if strand not in ["+", "-"]:
            raise ValueError(f"Invalid strand: {strand}. Expected '+' or '-'.")

        return {
            "gene_id": gene_id,
            "tss_index": tss_index,
            "chrom": chrom,
            "start": start,
            "end": end,
            "strand": strand,
            "selected_tss": start,
            "source": "manual_tss",
        }

    def _select_random_tss(self, gene_id):
        """
        TSS selection rule:

        1. If a gene has only one unique TSS:
           select that unique TSS.

        2. If a gene has more than one unique TSS:
           randomly select one TSS from unique_tss_list[1:],
           i.e. excluding the first TSS.

        This is designed for alternative-TSS sensitivity analysis.
        """
        if gene_id not in self.ensg2tss:
            raise KeyError(f"{gene_id} not found in ensg2tss")

        raw_tss_list = self._normalize_tss_list(self.ensg2tss[gene_id])
        unique_tss_list = self.dedup_tss(raw_tss_list)

        if len(unique_tss_list) == 0:
            raise ValueError(f"{gene_id} has no valid TSS")

        first_chrom, first_start, first_end, first_strand = unique_tss_list[0]
        first_tss = int(first_start)

        if len(unique_tss_list) == 1:
            selected_index = 0
            selected_tss_type = "only_tss"
        else:
            candidate_indices = list(range(1, len(unique_tss_list)))
            selected_index = self.rng.choice(candidate_indices)
            selected_tss_type = "alternative_tss_excluding_first"

        chrom, start, end, strand = unique_tss_list[selected_index]
        selected_tss = int(start)

        if chrom == first_chrom and strand == first_strand:
            distance_to_first_tss_bp = abs(selected_tss - first_tss)
        else:
            distance_to_first_tss_bp = np.nan

        return {
            "gene_id": gene_id,
            "chrom": chrom,
            "start": int(start),
            "end": int(end),
            "strand": strand,
            "selected_tss": selected_tss,
            "selected_tss_index": selected_index,
            "selected_tss_type": selected_tss_type,
            "first_chrom": first_chrom,
            "first_start": int(first_start),
            "first_end": int(first_end),
            "first_strand": first_strand,
            "first_tss": first_tss,
            "distance_to_first_tss_bp": distance_to_first_tss_bp,
            "n_tss_raw": len(raw_tss_list),
            "n_tss_unique": len(unique_tss_list),
        }

    def _read_one_mark(self, bw, chrom, tss):
        """
        Read one BigWig mark around selected TSS and return binned mean signal.

        Output shape:
            (n_bins,)
        """
        values = np.zeros(self.n_bins, dtype=np.float32)

        chrom_size = bw.chroms().get(chrom)
        if chrom_size is None:
            return values

        region_start = int(tss) - self.window

        for i in range(self.n_bins):
            bin_start = region_start + i * self.bin_size
            bin_end = bin_start + self.bin_size

            fetch_start = max(0, bin_start)
            fetch_end = min(chrom_size, bin_end)

            if fetch_end <= fetch_start:
                values[i] = 0.0
                continue

            v = bw.stats(
                chrom,
                fetch_start,
                fetch_end,
                type="mean",
                nBins=1,
            )[0]

            if v is None or np.isnan(v):
                v = 0.0

            values[i] = float(v)

        return values

    def _read_matrix_from_tss_meta(self, meta, strand_oriented=True):
        """
        Read histone matrix from a given TSS meta dictionary.

        Output:
            X_tss: shape = (n_bins, n_marks)
            meta: updated dict
        """
        chrom = meta["chrom"]
        tss = int(meta["selected_tss"])
        strand = meta.get("strand", "+")

        X_tss = np.zeros((self.n_bins, len(self.marks)), dtype=np.float32)

        for j, mark in enumerate(self.marks):
            X_tss[:, j] = self._read_one_mark(
                bw=self.bws[mark],
                chrom=chrom,
                tss=tss,
            )

        if self.log1p:
            X_tss = np.log1p(X_tss).astype(np.float32)

        if strand_oriented and strand == "-":
            X_tss = X_tss[::-1, :].copy()

        meta = dict(meta)
        meta.update(
            {
                "window": self.window,
                "bin_size": self.bin_size,
                "n_bins": self.n_bins,
                "n_marks": len(self.marks),
                "marks": ",".join(self.marks),
                "log1p": self.log1p,
                "strand_oriented": bool(strand_oriented),
                "matrix_shape": tuple(X_tss.shape),
            }
        )

        return X_tss, meta

    def get_tss_matrix(
        self,
        tss=None,
        chrom=None,
        start=None,
        end=None,
        strand=None,
        gene_id=None,
        tss_index=None,
        strand_oriented=True,
        add_batch_dim=True,
        return_meta=True,
    ):
        """
        Interface 2:
        Input one specific TSS and return preprocessed histone matrix.

        This output can be directly fed into a model.

        Parameters
        ----------
        tss:
            Flexible TSS input. Supported formats:

            1. pandas Series or dict from get_all_tss(...).iloc[i]
            2. tuple/list:
                (chrom, start, end, strand)
                or
                (chrom, tss, strand)
            3. None, with explicit chrom/start/end/strand arguments.

        chrom, start, end, strand:
            Explicit TSS fields. Used when tss is None.

        gene_id:
            Optional gene ID for metadata only.

        tss_index:
            Optional TSS index for metadata only.

        strand_oriented:
            If True, reverse bins for negative-strand TSSs so that matrices
            are aligned in transcription direction.

        add_batch_dim:
            If True, return shape = (1, n_bins, n_marks).
            If False, return shape = (n_bins, n_marks).

        return_meta:
            If True, return (X, meta).
            If False, return X only.

        Returns
        -------
        X:
            np.ndarray.
            If add_batch_dim=True:
                shape = (1, n_bins, n_marks)
            If add_batch_dim=False:
                shape = (n_bins, n_marks)

        meta:
            dict with TSS and preprocessing information.
        """
        meta = self._parse_tss_input(
            tss=tss,
            chrom=chrom,
            start=start,
            end=end,
            strand=strand,
            gene_id=gene_id,
            tss_index=tss_index,
        )

        X_tss, meta = self._read_matrix_from_tss_meta(
            meta,
            strand_oriented=strand_oriented,
        )

        if add_batch_dim:
            X_tss = X_tss[np.newaxis, :, :].astype(np.float32)
            meta["matrix_shape"] = tuple(X_tss.shape)

        if return_meta:
            return X_tss, meta

        return X_tss

    def _read_gene_matrix(self, gene_id, strand_oriented=True):
        """
        Return one gene matrix based on random TSS selection.

        Output:
            X_gene: shape = (n_bins, n_marks)
            meta: dict
        """
        meta = self._select_random_tss(gene_id)

        X_gene, meta = self._read_matrix_from_tss_meta(
            meta,
            strand_oriented=strand_oriented,
        )

        return X_gene, meta

    def get_gene_matrices(
        self,
        gene_ids=None,
        strand_oriented=True,
        skip_error=True,
        return_meta=True,
    ):
        """
        Unified interface for randomly selected TSS per gene.

        Parameters
        ----------
        gene_ids:
            None:
                process all genes in ensg2tss
            str:
                process one gene, automatically converted to [gene_id]
            list-like:
                process selected genes

        strand_oriented:
            If True, reverse bins for negative-strand genes so that all matrices
            are aligned in transcription direction.

        skip_error:
            If True, skip genes that fail. If False, raise error.

        return_meta:
            If True, return (X, meta_df). If False, return X only.

        Returns
        -------
        X:
            np.ndarray with shape = (n_genes, n_bins, n_marks)

        meta_df:
            pd.DataFrame with selected TSS information.
        """
        if gene_ids is None:
            gene_ids = list(self.ensg2tss.keys())
        elif isinstance(gene_ids, str):
            gene_ids = [gene_ids]
        else:
            gene_ids = list(gene_ids)

        X_list = []
        meta_list = []

        for gene_id in gene_ids:
            try:
                X_gene, meta = self._read_gene_matrix(
                    gene_id,
                    strand_oriented=strand_oriented,
                )
                X_list.append(X_gene)
                meta_list.append(meta)

            except Exception as e:
                if skip_error:
                    continue
                raise e

        if len(X_list) == 0:
            X = np.empty((0, self.n_bins, len(self.marks)), dtype=np.float32)
            meta_df = pd.DataFrame()
        else:
            X = np.stack(X_list, axis=0).astype(np.float32)
            meta_df = pd.DataFrame(meta_list)

        if return_meta:
            return X, meta_df

        return X

    def close(self):
        for bw in self.bws.values():
            bw.close()


if __name__ == "__main__":
    extractor = HistoneExtractor(
        ensg2tss_path="extra/datasets/annotations/ensg2tss.pickle",
        bigwig_dir="extra/datasets/epigenetic/hg19",
        eid="E003",
        seed=20240531,
        log1p=True,
    )

    gene_id = "ENSG00000000003"

    # ------------------------------------------------------------
    # Interface 1:
    # Return all TSSs for one gene
    # ------------------------------------------------------------
    tss_df = extractor.get_all_tss(
        gene_id,
        unique=True,
        as_dataframe=True,
    )

    print("All TSSs:")
    print(tss_df)

    # ------------------------------------------------------------
    # Interface 2:
    # Use one specific TSS to extract model-ready histone matrix
    # ------------------------------------------------------------
    one_tss = tss_df.iloc[0]

    X_tss, tss_meta = extractor.get_tss_matrix(
        one_tss,
        strand_oriented=True,
        add_batch_dim=True,
        return_meta=True,
    )

    print("Specific TSS matrix shape:")
    print(X_tss.shape)

    print("Specific TSS meta:")
    print(tss_meta)

    # ------------------------------------------------------------
    # Original interface:
    # Randomly select one TSS per gene
    # ------------------------------------------------------------
    gene_ids = [
        "ENSG00000022267",
        "ENSG00000000003",
    ]

    X, meta_df = extractor.get_gene_matrices(gene_ids)

    print("Random TSS matrices:")
    print(X.shape)
    print(meta_df)
    X_tss, meta = extractor.get_tss_matrix(
    chrom="chrX",
    start=99892101,
    end=99892102,
    strand="-",
    gene_id="ENSG00000000003",
    add_batch_dim=True,
    )
    print(meta)

    extractor.close()