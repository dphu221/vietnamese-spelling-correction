#!/usr/bin/env bash
# Launch training on Kaggle Notebook / Kaggle Script environment.
#
# Default paths:
#   Input Data: /kaggle/input/vietnamese-spelling-synthetic-1gb or auto-downloaded to /kaggle/working/data
#   Output:     /kaggle/working/checkpoints/full_1gb_bf16_3_epochs
#
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-/kaggle/working/checkpoints/full_1gb_bf16_3_epochs}"
AUTO_DOWNLOAD="${AUTO_DOWNLOAD:-1}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BUCKET_SIZE="${BUCKET_SIZE:-4096}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
AMP_DTYPE="${AMP_DTYPE:-auto}"

echo "=== Vietnamese Spelling Correction - Kaggle Runner ==="
echo "Output Directory: ${OUTPUT_DIR}"
echo "Auto Download:    ${AUTO_DOWNLOAD}"
echo "AMP Dtype:        ${AMP_DTYPE}"
echo "====================================================="

mkdir -p "${OUTPUT_DIR}"

FLAGS=()
if [[ "${AUTO_DOWNLOAD}" == "1" ]]; then
    FLAGS+=(--auto-download)
fi

python -u train_stream.py \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --bucket-size "${BUCKET_SIZE}" \
    --warmup-epochs "${WARMUP_EPOCHS}" \
    --amp-dtype "${AMP_DTYPE}" \
    --output-dir "${OUTPUT_DIR}" \
    "${FLAGS[@]}" \
    "$@"
