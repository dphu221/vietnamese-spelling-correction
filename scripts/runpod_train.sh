#!/usr/bin/env bash
# Launch the full-corpus BF16 trainer inside a RunPod Pod.
#
# Persistent volume layout expected by this script:
#   /workspace/data/train_full.jsonl
#   /workspace/data/validation_full.jsonl
#   /workspace/data/word_vocab_full.json
#   /workspace/data/char_vocab_full.json
#
# Training outputs remain on the same volume at /workspace/checkpoints/.
set -euo pipefail

DATA_DIR="${DATA_DIR:-/workspace/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/checkpoints/full_1gb_bf16_3_epochs}"
DATASET_REPO="${DATASET_REPO:-Sanng1112/vietnamese-spelling-synthetic-1gb}"
AUTO_DOWNLOAD="${AUTO_DOWNLOAD:-0}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BUCKET_SIZE="${BUCKET_SIZE:-4096}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"

required=(train_full.jsonl validation_full.jsonl word_vocab_full.json char_vocab_full.json)
missing=0
for filename in "${required[@]}"; do
    if [[ ! -f "${DATA_DIR}/${filename}" ]]; then
        printf 'Missing required dataset artifact: %s\n' "${DATA_DIR}/${filename}" >&2
        missing=1
    fi
done
if [[ "${missing}" -ne 0 ]]; then
    if [[ "${AUTO_DOWNLOAD}" != "1" ]]; then
        printf 'Missing data. Mount a volume at /workspace and either upload the artifacts or set AUTO_DOWNLOAD=1.\n' >&2
        exit 2
    fi
    printf 'Downloading public dataset %s to %s ...\n' "${DATASET_REPO}" "${DATA_DIR}"
    mkdir -p "${DATA_DIR}"
    hf download "${DATASET_REPO}" --repo-type dataset --local-dir "${DATA_DIR}"
    for filename in "${required[@]}"; do
        if [[ ! -f "${DATA_DIR}/${filename}" ]]; then
            printf 'Dataset download did not provide required artifact: %s\n' "${DATA_DIR}/${filename}" >&2
            exit 2
        fi
    done
fi

mkdir -p "${OUTPUT_DIR}"
exec python -u /workspace/app/train_stream.py \
    --train "${DATA_DIR}/train_full.jsonl" \
    --validation "${DATA_DIR}/validation_full.jsonl" \
    --word-vocab "${DATA_DIR}/word_vocab_full.json" \
    --char-vocab "${DATA_DIR}/char_vocab_full.json" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --bucket-size "${BUCKET_SIZE}" \
    --warmup-epochs "${WARMUP_EPOCHS}" \
    --amp-dtype "${AMP_DTYPE}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
