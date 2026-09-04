#!/usr/bin/env bash
set -euo pipefail

export WANDB_MODE=offline

META_DIR="benchmarks/revised_v2/reviewer3/meta_datasets"
NPY_DIR="/GPUFS/sysu_jjzhang_3/wanghan/burstformer/extra/datasets/processed/v1"

mkdir -p logs benchmarks/revised_v2/reviewer3/checkpoints

run_one () {
    META_FILE=$1
    OUT_TAG=$2

    for fold in 0 1 2 3
    do
        echo "Running ${OUT_TAG}, fold=${fold}"
        echo "Meta file: ${META_FILE}"

        python src/model/train.py \
            --config configs/default.yaml \
            --meta "${META_FILE}" \
            --npy-dir "${NPY_DIR}" \
            --fold "${fold}" \
            -o "benchmarks/revised_v2/reviewer3/checkpoints/${OUT_TAG}.fold.${fold}.bs_bf_para.model.pt" \
            --exp-id 2 \
            --binsizes 500 \
            > "logs/${OUT_TAG}.fold.${fold}.bs_bf_para.train.log" 2>&1

        sleep 20
    done
}

echo "============================================================"
echo "Training models using txburst-derived labels"
echo "============================================================"


# run_one "${META_DIR}/meta_data_E003_delay_1.0_with_cellsize_1_threshold_mean.csv"  "E003.threshold.txburst"
# run_one "${META_DIR}/meta_data_E116_delay_1.0_with_cellsize_1_threshold_mean.csv"  "E116.threshold.txburst"
# run_one "${META_DIR}/meta_data_E118_delay_1.0_with_cellsize_1_threshold_mean.csv"  "E118.threshold.txburst"

run_one "${META_DIR}/meta_data_E003_delay_1.0_with_cellsize_1_threshold_median_exclude_mid20.csv"  "E003.threshold_median_exclude_mid20.txburst"
run_one "${META_DIR}/meta_data_E116_delay_1.0_with_cellsize_1_threshold_median_exclude_mid20.csv"  "E116.threshold_median_exclude_mid20.txburst"
run_one "${META_DIR}/meta_data_E118_delay_1.0_with_cellsize_1_threshold_median_exclude_mid20.csv"  "E118.threshold_median_exclude_mid20.txburst"