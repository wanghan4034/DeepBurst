#!/usr/bin/env python3
import argparse
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--refgene", required=True)
    p.add_argument("--chrom-sizes", required=True)
    p.add_argument("--out-bed", required=True)
    p.add_argument("--nm-only", action="store_true")
    return p.parse_args()


def load_chrom_sizes(path):
    sizes = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                chrom, size = line.rstrip("\n").split("\t")[:2]
                sizes[chrom] = int(size)
    return sizes


def main():
    args = parse_args()
    chrom_sizes = load_chrom_sizes(args.chrom_sizes)

    cols = [
        "bin", "name", "chrom", "strand", "txStart", "txEnd",
        "cdsStart", "cdsEnd", "exonCount", "exonStarts", "exonEnds",
        "score", "name2", "cdsStartStat", "cdsEndStat", "exonFrames"
    ]

    df = pd.read_csv(
        args.refgene,
        sep="\t",
        header=None,
        names=cols,
        compression="gzip" if args.refgene.endswith(".gz") else None,
    )

    df = df[df["chrom"].isin(chrom_sizes)].copy()
    df = df[df["strand"].isin(["+", "-"])].copy()

    if args.nm_only:
        df = df[df["name"].astype(str).str.startswith("NM_")].copy()

    df["txStart"] = df["txStart"].astype(int)
    df["txEnd"] = df["txEnd"].astype(int)
    df["cdsStart"] = df["cdsStart"].astype(int)
    df["cdsEnd"] = df["cdsEnd"].astype(int)

    with open(args.out_bed, "w") as out:
        out.write(
            'track name="mm9 RefSeq NM" '
            'description="UCSC mm9 RefSeq NM transcripts" '
            'visibility=2 itemRgb="On"\n'
        )

        for _, r in df.iterrows():
            chrom = str(r["chrom"])
            chrom_size = chrom_sizes[chrom]

            tx_start = int(r["txStart"])
            tx_end = int(r["txEnd"])
            cds_start = int(r["cdsStart"])
            cds_end = int(r["cdsEnd"])

            if tx_start < 0 or tx_end > chrom_size or tx_end <= tx_start:
                continue

            exon_starts = [
                int(x) for x in str(r["exonStarts"]).rstrip(",").split(",")
                if x != ""
            ]
            exon_ends = [
                int(x) for x in str(r["exonEnds"]).rstrip(",").split(",")
                if x != ""
            ]

            if len(exon_starts) != len(exon_ends):
                continue

            block_sizes = []
            block_starts = []
            valid = True

            for s, e in zip(exon_starts, exon_ends):
                if e <= s:
                    valid = False
                    break
                if s < tx_start or e > tx_end:
                    valid = False
                    break

                block_sizes.append(e - s)
                block_starts.append(s - tx_start)

            if not valid or len(block_sizes) == 0:
                continue

            block_count = len(block_sizes)

            gene_name = str(r["name2"]).replace("|", "_").replace(" ", "_")
            transcript_id = str(r["name"]).replace("|", "_").replace(" ", "_")
            bed_name = f"{gene_name}_{transcript_id}"

            # thickStart / thickEnd 必须落在 transcript 区间内
            thick_start = max(tx_start, min(cds_start, tx_end))
            thick_end = max(thick_start, min(cds_end, tx_end))

            out.write(
                "\t".join([
                    chrom,
                    str(tx_start),
                    str(tx_end),
                    bed_name,
                    "0",
                    str(r["strand"]),
                    str(thick_start),
                    str(thick_end),
                    "0,0,255",
                    str(block_count),
                    ",".join(map(str, block_sizes)) + ",",
                    ",".join(map(str, block_starts)) + ",",
                ]) + "\n"
            )


if __name__ == "__main__":
    main()