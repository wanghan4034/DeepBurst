#!/usr/bin/env python3
"""
Generate mm9 gene-level TSS BED / promoter BED files and an ENSMUSG-to-TSS
pickle file from UCSC refGene plus GENCODE/Ensembl gene annotation.

Default output pickle:
    refGene.first_transcript_per_gene.ensm2TSS.pkl

Default pickle structure when --pickle-format tuple:
    {
        "ENSMUSG00000000001": ("chr3", 108014595, 108014596, "-"),
        ...
    }

The TSS interval is a UCSC/BED-style 0-based, half-open, one-base interval:
    + strand: [txStart, txStart + 1)
    - strand: [txEnd - 1, txEnd)
"""

import argparse
import os
import pickle
import urllib.request
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple, Union

import pandas as pd


REFGENE_COLS = [
    "bin",
    "name",
    "chrom",
    "strand",
    "txStart",
    "txEnd",
    "cdsStart",
    "cdsEnd",
    "exonCount",
    "exonStarts",
    "exonEnds",
    "score",
    "name2",
    "cdsStartStat",
    "cdsEndStat",
    "exonFrames",
]

GTF_COLS = [
    "chrom",
    "source",
    "feature",
    "start",
    "end",
    "score",
    "strand",
    "frame",
    "attribute",
]


TSSValue = Union[Tuple[str, int, int, str], Dict[str, object]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate mm9 gene-level TSS BED, promoter BED, and ensm2TSS pickle. "
            "Representative transcript is selected by lexicographic RefSeq transcript ID order. "
            "Ensembl gene ID is assigned after TSS selection."
        )
    )

    p.add_argument("--mm9-dir", default="extra/datasets/annotations/mm9")
    p.add_argument("--refgene", default=None)
    p.add_argument("--chrom-sizes", default=None)
    p.add_argument("--known-to-refseq", default=None)
    p.add_argument("--known-to-ensembl", default=None)
    p.add_argument("--ensembl-gtf", default=None)

    p.add_argument("--out-tss-bed", default=None)
    p.add_argument("--out-promoter-bed", default=None)
    p.add_argument("--out-ensm2tss-pkl", default=None)
    p.add_argument("--out-ensm2tss-tsv", default=None)

    p.add_argument("--upstream", type=int, default=20000)
    p.add_argument("--downstream", type=int, default=20000)
    p.add_argument(
        "--mode",
        choices=["first_transcript_per_gene", "all_transcripts"],
        default="all_transcripts",
        help=(
            "first_transcript_per_gene: one representative TSS per gene symbol; "
            "all_transcripts: retain every NM_ RefSeq transcript."
        ),
    )
    p.add_argument(
        "--pickle-format",
        choices=["tuple", "dict"],
        default="tuple",
        help=(
            "tuple: ENSMUSG -> (chrom, tss_start, tss_end, strand); "
            "dict: ENSMUSG -> metadata dict. In all_transcripts mode, values are lists."
        ),
    )
    p.add_argument(
        "--keep-missing-ensembl",
        action="store_true",
        help="Also keep records without ENSMUSG using keys NA|gene_name. Default: skip them in pickle.",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not download missing annotation files; fail if required inputs are missing.",
    )
    return p.parse_args()


def compression_of(path: str):
    return "gzip" if path.endswith(".gz") else None


def mkdir_for_file(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def download_if_missing(url: str, out_path: str, skip_download: bool = False) -> None:
    mkdir_for_file(out_path)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"[SKIP] Existing file: {out_path}")
        return

    if skip_download:
        raise FileNotFoundError(f"Missing required file and --skip-download was set: {out_path}")

    print(f"[DOWNLOAD] {url}")
    print(f"[TO] {out_path}")
    urllib.request.urlretrieve(url, out_path)

    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError(f"Download failed or empty file: {out_path}")


def resolve_paths(args: argparse.Namespace) -> Dict[str, str]:
    mm9_dir = args.mm9_dir
    os.makedirs(mm9_dir, exist_ok=True)

    mode = args.mode
    paths = {
        "refgene": args.refgene or os.path.join(mm9_dir, "refGene.txt.gz"),
        # chromInfo.txt.gz has at least chrom and size columns, so it can be parsed like *.sizes.
        "chrom_sizes": args.chrom_sizes or os.path.join(mm9_dir, "chromInfo.txt.gz"),
        "known_to_refseq": args.known_to_refseq or os.path.join(mm9_dir, "knownToRefSeq.txt.gz"),
        "known_to_ensembl": args.known_to_ensembl or os.path.join(mm9_dir, "knownToEnsembl.txt.gz"),
        # GENCODE vM1 is mm9/NCBIM37-era annotation and contains ENSMUSG IDs.
        "ensembl_gtf": args.ensembl_gtf or os.path.join(mm9_dir, "gencode.vM1.annotation.gtf.gz"),
        "out_tss": args.out_tss_bed or os.path.join(mm9_dir, f"refGene.{mode}.tss.bed"),
        "out_promoter": args.out_promoter_bed or os.path.join(mm9_dir, f"refGene.{mode}.promoter_40kb.bed"),
        "out_ensm2tss_pkl": args.out_ensm2tss_pkl or os.path.join(mm9_dir, f"refGene.{mode}.ensm2TSS.pkl"),
        "out_ensm2tss_tsv": args.out_ensm2tss_tsv or os.path.join(mm9_dir, f"refGene.{mode}.ensm2TSS.tsv"),
    }
    return paths


def download_required_inputs(paths: Dict[str, str], skip_download: bool = False) -> None:
    download_if_missing(
        "https://hgdownload.soe.ucsc.edu/goldenPath/mm9/database/refGene.txt.gz",
        paths["refgene"],
        skip_download=skip_download,
    )
    download_if_missing(
        "https://hgdownload.soe.ucsc.edu/goldenPath/mm9/database/chromInfo.txt.gz",
        paths["chrom_sizes"],
        skip_download=skip_download,
    )
    download_if_missing(
        "https://hgdownload.soe.ucsc.edu/goldenPath/mm9/database/knownToRefSeq.txt.gz",
        paths["known_to_refseq"],
        skip_download=skip_download,
    )
    download_if_missing(
        "https://hgdownload.soe.ucsc.edu/goldenPath/mm9/database/knownToEnsembl.txt.gz",
        paths["known_to_ensembl"],
        skip_download=skip_download,
    )
    download_if_missing(
        "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M1/gencode.vM1.annotation.gtf.gz",
        paths["ensembl_gtf"],
        skip_download=skip_download,
    )


def load_chrom_sizes(path: str) -> Dict[str, int]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise FileNotFoundError(f"Missing chrom sizes file: {path}")

    sizes: Dict[str, int] = {}
    with pd.read_csv(
        path,
        sep="\t",
        header=None,
        compression=compression_of(path),
        chunksize=200000,
        dtype=str,
    ) as reader:
        for chunk in reader:
            if chunk.shape[1] < 2:
                raise ValueError(f"Chrom sizes file must contain at least two columns: {path}")
            for chrom, size in zip(chunk.iloc[:, 0], chunk.iloc[:, 1]):
                if pd.isna(chrom) or pd.isna(size):
                    continue
                sizes[str(chrom)] = int(size)

    if not sizes:
        raise RuntimeError(f"No chromosome sizes were parsed from: {path}")
    return sizes


def parse_gtf_attributes(attr: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in str(attr).strip().split(";"):
        item = item.strip()
        if not item or " " not in item:
            continue
        key, value = item.split(" ", 1)
        out[key] = value.strip().strip('"')
    return out


def load_gene_name_to_ensembl_gene_id_from_gtf(gtf_path: str) -> Dict[str, str]:
    """
    Build a reproducible gene_name -> Ensembl gene ID mapping from GENCODE/Ensembl GTF.
    If the same gene_name maps to multiple ENSMUSG IDs, keep the lexicographically first ID.
    """
    if not os.path.exists(gtf_path) or os.path.getsize(gtf_path) == 0:
        raise FileNotFoundError(f"Missing GTF file: {gtf_path}")

    records: List[Tuple[str, str]] = []
    for chunk in pd.read_csv(
        gtf_path,
        sep="\t",
        comment="#",
        header=None,
        names=GTF_COLS,
        compression=compression_of(gtf_path),
        chunksize=200000,
        dtype=str,
    ):
        chunk = chunk[chunk["feature"] == "gene"]
        if chunk.empty:
            continue

        for attr in chunk["attribute"]:
            parsed = parse_gtf_attributes(attr)
            gene_id = parsed.get("gene_id", "").split(".")[0]
            gene_name = parsed.get("gene_name", "")
            if gene_id.startswith("ENSMUSG") and gene_name:
                records.append((gene_name, gene_id))

    if not records:
        raise RuntimeError(f"No ENSMUSG gene_id records found in GTF: {gtf_path}")

    df = pd.DataFrame(records, columns=["gene_name", "ensembl_gene_id"])
    df = df.drop_duplicates().sort_values(["gene_name", "ensembl_gene_id"])
    df = df.drop_duplicates(subset=["gene_name"], keep="first")
    return dict(zip(df["gene_name"], df["ensembl_gene_id"]))


def load_refseq_to_ensembl_transcript(known_to_refseq: str, known_to_ensembl: str) -> Dict[str, str]:
    """
    Build auxiliary RefSeq transcript ID -> Ensembl transcript ID mapping.
    This does not determine gene_id or representative TSS selection.
    """
    for path in [known_to_refseq, known_to_ensembl]:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            raise FileNotFoundError(f"Missing transcript mapping file: {path}")

    k2r = pd.read_csv(
        known_to_refseq,
        sep="\t",
        header=None,
        names=["known_id", "refseq_transcript_id"],
        compression=compression_of(known_to_refseq),
        dtype=str,
    )
    k2e = pd.read_csv(
        known_to_ensembl,
        sep="\t",
        header=None,
        names=["known_id", "ensembl_transcript_id"],
        compression=compression_of(known_to_ensembl),
        dtype=str,
    )

    merged = k2r.merge(k2e, on="known_id", how="inner")
    merged = merged.dropna(subset=["refseq_transcript_id", "ensembl_transcript_id"])
    merged = merged.sort_values(["refseq_transcript_id", "ensembl_transcript_id"])
    merged = merged.drop_duplicates(subset=["refseq_transcript_id"], keep="first")
    return dict(zip(merged["refseq_transcript_id"], merged["ensembl_transcript_id"]))


def load_and_select_refgene(refgene_path: str, chrom_sizes: Dict[str, int], mode: str) -> Tuple[pd.DataFrame, int, int, int]:
    df = pd.read_csv(
        refgene_path,
        sep="\t",
        header=None,
        names=REFGENE_COLS,
        compression=compression_of(refgene_path),
        dtype={"name": str, "name2": str, "chrom": str, "strand": str},
    )
    n_raw = df.shape[0]

    # UCSC refGene:
    #   name  = RefSeq transcript ID, e.g. NM_013633
    #   name2 = gene symbol, e.g. Pou5f1
    df = df[df["chrom"].isin(chrom_sizes)].copy()
    df = df[df["strand"].isin(["+", "-"])].copy()
    df = df[df["name"].astype(str).str.startswith("NM_")].copy()
    df["txStart"] = df["txStart"].astype(int)
    df["txEnd"] = df["txEnd"].astype(int)
    df["refseq_transcript_id"] = df["name"].astype(str)
    df["gene_name"] = df["name2"].astype(str)
    n_after_filter = df.shape[0]

    # Critical selection rule:
    # 1. sort by gene_name;
    # 2. within each gene_name, sort by RefSeq transcript ID lexicographically;
    # 3. retain the first transcript when mode == first_transcript_per_gene.
    df = df.sort_values(["gene_name", "refseq_transcript_id"]).copy()
    if mode == "first_transcript_per_gene":
        df = df.drop_duplicates(subset=["gene_name"], keep="first").copy()
    n_after_selection = df.shape[0]

    return df, n_raw, n_after_filter, n_after_selection


def add_gene_and_transcript_ids(
    df: pd.DataFrame,
    gene_name_to_ensembl_gene_id: Dict[str, str],
    refseq_to_ensembl_transcript: Dict[str, str],
) -> pd.DataFrame:
    df = df.copy()
    df["ensembl_gene_id"] = df["gene_name"].map(gene_name_to_ensembl_gene_id).fillna("NA").astype(str)
    df["gene_id"] = df["ensembl_gene_id"]
    df["ensembl_transcript_id"] = (
        df["refseq_transcript_id"].map(refseq_to_ensembl_transcript).fillna("NA").astype(str)
    )
    return df


def compute_tss_interval(row: pd.Series, chrom_sizes: Dict[str, int]) -> Tuple[int, int]:
    chrom = str(row["chrom"])
    strand = str(row["strand"])
    chrom_size = chrom_sizes[chrom]

    if strand == "+":
        tss = int(row["txStart"])
    else:
        # UCSC/BED half-open coordinate: negative-strand TSS is [txEnd - 1, txEnd)
        tss = int(row["txEnd"]) - 1

    if tss < 0 or tss >= chrom_size:
        raise ValueError(f"TSS out of bounds: {chrom}:{tss}, chrom_size={chrom_size}")

    return tss, tss + 1


def make_tss_record(row: pd.Series, tss_start: int, tss_end: int) -> Dict[str, object]:
    return {
        "chrom": str(row["chrom"]),
        "start": int(tss_start),
        "end": int(tss_end),
        "tss": int(tss_start),
        "strand": str(row["strand"]),
        "gene_id": str(row["gene_id"]),
        "ensembl_gene_id": str(row["ensembl_gene_id"]),
        "gene_name": str(row["gene_name"]),
        "refseq_transcript_id": str(row["refseq_transcript_id"]),
        "ensembl_transcript_id": str(row["ensembl_transcript_id"]),
        "txStart": int(row["txStart"]),
        "txEnd": int(row["txEnd"]),
    }


def record_to_pickle_value(record: Dict[str, object], pickle_format: str) -> TSSValue:
    if pickle_format == "tuple":
        return (
            str(record["chrom"]),
            int(record["start"]),
            int(record["end"]),
            str(record["strand"]),
        )
    if pickle_format == "dict":
        return dict(record)
    raise ValueError(f"Unsupported pickle format: {pickle_format}")


def write_outputs(
    df: pd.DataFrame,
    chrom_sizes: Dict[str, int],
    out_tss: str,
    out_promoter: str,
    out_ensm2tss_pkl: str,
    out_ensm2tss_tsv: str,
    upstream: int,
    downstream: int,
    mode: str,
    pickle_format: str,
    keep_missing_ensembl: bool,
) -> Dict[str, int]:
    for path in [out_tss, out_promoter, out_ensm2tss_pkl, out_ensm2tss_tsv]:
        mkdir_for_file(path)

    rows_for_tsv: List[Dict[str, object]] = []
    ensm2tss_first: Dict[str, TSSValue] = {}
    ensm2tss_all: Dict[str, List[TSSValue]] = defaultdict(list)

    n_written = 0
    n_pickle_records = 0
    n_tss_out_of_bounds = 0
    n_missing_ensembl_gene = int((df["ensembl_gene_id"] == "NA").sum())
    n_missing_ensembl_transcript = int((df["ensembl_transcript_id"] == "NA").sum())
    n_duplicate_ensembl_gene_in_pickle = 0

    with open(out_tss, "w") as tss_out, open(out_promoter, "w") as prom_out:
        for _, row in df.iterrows():
            chrom = str(row["chrom"])
            strand = str(row["strand"])
            chrom_size = chrom_sizes[chrom]
            gene_id = str(row["gene_id"])
            gene_name = str(row["gene_name"])
            refseq_transcript_id = str(row["refseq_transcript_id"])
            ensembl_gene_id = str(row["ensembl_gene_id"])
            ensembl_transcript_id = str(row["ensembl_transcript_id"])

            try:
                tss_start, tss_end = compute_tss_interval(row, chrom_sizes)
            except ValueError:
                n_tss_out_of_bounds += 1
                continue

            record_name = f"{gene_name}|{gene_id}"
            bed_extra = (
                f"{gene_id}\t{gene_name}\t{refseq_transcript_id}\t"
                f"{ensembl_gene_id}\t{ensembl_transcript_id}"
            )
            tss_out.write(
                f"{chrom}\t{tss_start}\t{tss_end}\t{record_name}\t0\t{strand}\t{bed_extra}\n"
            )

            prom_start = max(0, tss_start - upstream)
            prom_end = min(chrom_size, tss_start + downstream)
            if prom_end > prom_start:
                prom_out.write(
                    f"{chrom}\t{prom_start}\t{prom_end}\t{record_name}\t0\t{strand}\t{bed_extra}\n"
                )

            record = make_tss_record(row, tss_start, tss_end)
            record["promoter_start"] = int(prom_start)
            record["promoter_end"] = int(prom_end)
            rows_for_tsv.append(record)

            if gene_id == "NA":
                if not keep_missing_ensembl:
                    n_written += 1
                    continue
                pickle_key = f"NA|{gene_name}"
            else:
                pickle_key = gene_id

            value = record_to_pickle_value(record, pickle_format)
            if mode == "all_transcripts":
                ensm2tss_all[pickle_key].append(value)
                n_pickle_records += 1
            else:
                if pickle_key in ensm2tss_first:
                    n_duplicate_ensembl_gene_in_pickle += 1
                else:
                    ensm2tss_first[pickle_key] = value
                    n_pickle_records += 1

            n_written += 1

    ensm2tss: Union[Dict[str, TSSValue], Dict[str, List[TSSValue]]]
    if mode == "all_transcripts":
        ensm2tss = dict(ensm2tss_all)
    else:
        ensm2tss = ensm2tss_first

    with open(out_ensm2tss_pkl, "wb") as f:
        pickle.dump(ensm2tss, f, protocol=pickle.HIGHEST_PROTOCOL)

    pd.DataFrame(rows_for_tsv).to_csv(out_ensm2tss_tsv, sep="\t", index=False)

    return {
        "records_written_to_bed": n_written,
        "records_in_pickle": len(ensm2tss),
        "pickle_tss_entries": n_pickle_records,
        "missing_ensembl_gene_id": n_missing_ensembl_gene,
        "missing_ensembl_transcript_id": n_missing_ensembl_transcript,
        "duplicate_ensembl_gene_skipped_in_pickle": n_duplicate_ensembl_gene_in_pickle,
        "tss_out_of_bounds_skipped": n_tss_out_of_bounds,
    }


def preview_pickle(path: str, n: int = 5) -> None:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"[Pickle preview] {path}")
    print(f"Type: {type(obj).__name__}; entries: {len(obj)}")
    for i, (key, value) in enumerate(obj.items()):
        if i >= n:
            break
        print(f"  {key}: {value}")


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)

    download_required_inputs(paths, skip_download=args.skip_download)

    chrom_sizes = load_chrom_sizes(paths["chrom_sizes"])
    gene_name_to_ensembl_gene_id = load_gene_name_to_ensembl_gene_id_from_gtf(paths["ensembl_gtf"])
    refseq_to_ensembl_transcript = load_refseq_to_ensembl_transcript(
        paths["known_to_refseq"],
        paths["known_to_ensembl"],
    )

    df, n_raw, n_after_filter, n_after_selection = load_and_select_refgene(
        paths["refgene"], chrom_sizes, args.mode
    )
    df = add_gene_and_transcript_ids(
        df,
        gene_name_to_ensembl_gene_id,
        refseq_to_ensembl_transcript,
    )

    stats = write_outputs(
        df=df,
        chrom_sizes=chrom_sizes,
        out_tss=paths["out_tss"],
        out_promoter=paths["out_promoter"],
        out_ensm2tss_pkl=paths["out_ensm2tss_pkl"],
        out_ensm2tss_tsv=paths["out_ensm2tss_tsv"],
        upstream=args.upstream,
        downstream=args.downstream,
        mode=args.mode,
        pickle_format=args.pickle_format,
        keep_missing_ensembl=args.keep_missing_ensembl,
    )

    print("[DONE]")
    print(f"Raw refGene records:                       {n_raw}")
    print(f"After chrom/strand/NM_ filtering:           {n_after_filter}")
    print(f"After representative transcript selection:  {n_after_selection}")
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').capitalize():42s} {value}")
    print()
    print(f"TSS BED:        {paths['out_tss']}")
    print(f"Promoter BED:   {paths['out_promoter']}")
    print(f"ensm2TSS pickle:{paths['out_ensm2tss_pkl']}")
    print(f"ensm2TSS TSV:   {paths['out_ensm2tss_tsv']}")
    print()
    print("[Selection rule]")
    print("TSS anchor is selected by gene_name + RefSeq transcript ID lexicographic order.")
    print("Ensembl gene ID is assigned after TSS selection and used as the pickle key.")
    print("Default pickle value is (chrom, tss_start, tss_end, strand).")
    print()
    preview_pickle(paths["out_ensm2tss_pkl"])


if __name__ == "__main__":
    main()
