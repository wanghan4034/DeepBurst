#!/usr/bin/env bash
set -euo pipefail

MM9_DIR="extra/datasets/annotations/mm9"

REFGENE="${MM9_DIR}/refGene.txt.gz"
CHROM_SIZES="${MM9_DIR}/mm9.fa.sizes"

OUT_TSS="${MM9_DIR}/refGene.first_transcript_per_gene.tss.bed"
OUT_PROMOTER="${MM9_DIR}/refGene.first_transcript_per_gene.promoter_40kb.bed"

SCRIPT="benchmarks/revised_v2/reviewer3/scripts/make_tss_from_ucsc_refgene.py"

mkdir -p "${MM9_DIR}"

# 下载 mm9 refGene
if [ ! -s "${REFGENE}" ]; then
    wget -c -O "${REFGENE}" \
        "https://hgdownload.soe.ucsc.edu/goldenPath/mm9/database/refGene.txt.gz"
fi

# 检查 chrom sizes
if [ ! -s "${CHROM_SIZES}" ]; then
    echo "[ERROR] Missing ${CHROM_SIZES}"
    echo "Please generate it first:"
    echo "samtools faidx extra/datasets/annotations/mm9/mm9.fa"
    echo "cut -f1,2 extra/datasets/annotations/mm9/mm9.fa.fai > ${CHROM_SIZES}"
    exit 1
fi

# 运行 Python 脚本
python "${SCRIPT}" \
    --refgene "${REFGENE}" \
    --chrom-sizes "${CHROM_SIZES}" \
    --out-tss-bed "${OUT_TSS}" \
    --out-promoter-bed "${OUT_PROMOTER}" \
    --mode first_transcript_per_gene \
    --upstream 20000 \
    --downstream 20000

# 检查输出
ls -lh "${OUT_TSS}" "${OUT_PROMOTER}"

echo "[DONE]"
echo "${OUT_TSS}"
echo "${OUT_PROMOTER}"