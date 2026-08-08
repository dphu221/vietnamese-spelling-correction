# Pinned to the PyTorch/CUDA combination used and smoke-tested locally.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /workspace/app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace/app

# ``hf`` lets a RunPod member populate an empty persistent volume directly
# from the public dataset repository before training starts.
RUN python -m pip install --no-cache-dir huggingface_hub==1.27.0

COPY models.py train_stream.py ./
COPY datasets/build_dataset.py datasets/README.md ./datasets/
COPY scripts/runpod_train.sh ./scripts/runpod_train.sh

RUN chmod 0755 ./scripts/runpod_train.sh

# By convention, mount a persistent RunPod volume at /workspace and place the
# four processed training artifacts under /workspace/data. The entrypoint
# writes histories and checkpoints below /workspace/checkpoints.
ENTRYPOINT ["/workspace/app/scripts/runpod_train.sh"]
