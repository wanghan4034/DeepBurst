#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Download processed matrices from EMBL-EBI Single Cell Expression Atlas
# Larsson et al. 2019 Nature:
# "Genomic encoding of transcriptional burst kinetics"
#
# Target:
#   E-MTAB-6362
#   E-MTAB-7098
#
# Downloaded:
#   1. project.h5ad
#   2. aggregated filtered raw count matrix: mtx.gz + rows.gz + cols.gz
#   3. aggregated filtered normalised count matrix: mtx.gz + rows.gz + cols.gz
#   4. expression TPM matrix: mtx.gz + rows.gz + cols.gz
#   5. cell metadata / SDRF / IDF
# ============================================================

OUTDIR="extra/datasets/mouse/larsson2019_expression_atlas_processed"
LOGDIR="${OUTDIR}/logs"
mkdir -p "${OUTDIR}" "${LOGDIR}"

ACCESSIONS=(
  "E-MTAB-6362"
  "E-MTAB-7098"
)

BASE_URL="https://ftp.ebi.ac.uk/pub/databases/microarray/data/atlas/sc_experiments"

check_command() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "[ERROR] Required command not found: ${cmd}" >&2
        exit 1
    fi
}

check_command wget

download_one_file() {
    local url="$1"
    local out="$2"

    mkdir -p "$(dirname "${out}")"

    if [[ -s "${out}" ]]; then
        echo "[SKIP] Existing: ${out}"
        return
    fi

    echo "[DOWNLOAD] ${url}"
    wget -c -O "${out}" "${url}"
}

download_one_accession() {
    local acc="$1"
    local out="${OUTDIR}/${acc}"
    local url="${BASE_URL}/${acc}"

    mkdir -p "${out}"

    echo "============================================================"
    echo "[ACCESSION] ${acc}"
    echo "[URL] ${url}/"
    echo "[OUT] ${out}"
    echo "============================================================"

    local files=(
      "${acc}.project.h5ad"

      "${acc}.aggregated_filtered_counts.mtx.gz"
      "${acc}.aggregated_filtered_counts.mtx_rows.gz"
      "${acc}.aggregated_filtered_counts.mtx_cols.gz"

      "${acc}.aggregated_filtered_normalised_counts.mtx.gz"
      "${acc}.aggregated_filtered_normalised_counts.mtx_rows.gz"
      "${acc}.aggregated_filtered_normalised_counts.mtx_cols.gz"

      "${acc}.expression_tpm.mtx.gz"
      "${acc}.expression_tpm.mtx_rows.gz"
      "${acc}.expression_tpm.mtx_cols.gz"

      "${acc}.cell_metadata.tsv"
      "${acc}.clusters.tsv"
      "${acc}.condensed-sdrf.tsv"
      "${acc}.idf.txt"
      "${acc}.sdrf.txt"
      "${acc}.software.tsv"
    )

    for fname in "${files[@]}"; do
        download_one_file "${url}/${fname}" "${out}/${fname}" \
          2>&1 | tee -a "${LOGDIR}/${acc}.download.log"
    done

    echo "[DONE] ${acc}"
    echo
}

for acc in "${ACCESSIONS[@]}"; do
    download_one_accession "${acc}"
done

echo "============================================================"
echo "[ALL DONE]"
echo "Downloaded files:"
find "${OUTDIR}" -type f \
  ! -name "._*" \
  ! -name "robots.txt" \
  | sort
echo "============================================================"