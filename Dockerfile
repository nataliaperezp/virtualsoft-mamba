# =============================================================================
# Dockerfile — Mamba SSM Training Job para Vertex AI
# Base: CUDA 12.1 devel (necesario para compilar mamba-ssm y causal-conv1d)
# =============================================================================
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# ---------------------------------------------------------------------------
# Sistema
# ---------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        curl \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python

# ---------------------------------------------------------------------------
# uv
# ---------------------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1

# ---------------------------------------------------------------------------
# 1. PyTorch con CUDA 11.8
# ---------------------------------------------------------------------------
RUN uv pip install --system torch==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu118

# ---------------------------------------------------------------------------
# 2. mamba-ssm y causal-conv1d (requieren compilación con nvcc)
# ---------------------------------------------------------------------------
RUN uv pip install --system \
    causal-conv1d==1.4.0 \
    mamba-ssm==2.2.2

# ---------------------------------------------------------------------------
# 3. Resto de dependencias
# ---------------------------------------------------------------------------
RUN uv pip install --system \
    polars==1.39.3 \
    numpy==2.2.6 \
    scikit-learn==1.7.2 \
    pyarrow==23.0.1 \
    wandb==0.25.1 \
    optuna==4.8.0 \
    gcsfs==2026.1.0 \
    fsspec==2026.2.0 \
    google-cloud-storage==3.10.1 \
    psutil==5.9.3 \
    tqdm==4.67.3 \
    python-dotenv==1.2.2

# ---------------------------------------------------------------------------
# Código fuente
# ---------------------------------------------------------------------------
WORKDIR /app
COPY Mamba.py .
COPY split_estratificado_uids.json .

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
CMD ["python", "Mamba.py"]
