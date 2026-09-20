# Runs the truetype.ai replica API. CPU only by default. For an NVIDIA GPU, build
# against the CUDA wheel index that matches your driver (see
# https://pytorch.org/get-started/locally/ for the current one, e.g. cu128) and run
# with --gpus all:
#
#   docker build \
#     --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
#     -t truetype-replica .
#   docker run --rm --gpus all -p 8000:8000 \
#     -e TYPESAFE_REPLICA_DEVICE=cuda \
#     -v ~/.cache/huggingface:/home/app/.cache/huggingface \
#     truetype-replica
#
# The 22GB Gemma weights are not baked in. Mount a Hugging Face cache (or a named
# volume) at HF_HOME and they download on first start.
FROM python:3.13-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

# Install torch separately so one build argument selects CPU or CUDA wheels.
# accelerate supports device_map and low_cpu_mem_usage; pillow supports the
# multimodal auto-classes.
RUN pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" torch \
    && pip install --no-cache-dir transformers fastapi uvicorn pydantic accelerate pillow

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps -e .

RUN useradd --create-home --uid 10001 app \
    && mkdir -p "${HF_HOME}" \
    && chown -R app:app /app "${HF_HOME}"
USER app

EXPOSE 8000

# Model loading can take several minutes. The healthcheck waits 10 minutes before
# polling /health every 30 seconds.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).read()"

CMD ["truetype-api"]
