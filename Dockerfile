# =============================================================================
# Dockerfile — Mamba SSM Training Job para Vertex AI
# Base: CUDA 11.8 devel (alineado con torch cu118 probado en local)
# =============================================================================
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# ---------------------------------------------------------------------------
# Sistema
# ---------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        curl \
        git \
        build-essential \
        tzdata \
    && rm -rf /var/lib/apt/lists/*
ENV TZ=UTC

# ---------------------------------------------------------------------------
# Fijar python3.11 como el intérprete por defecto para TODO (python, pip)
# Usamos get-pip.py para obtener pip asociado a python3.11, no al 3.10 del SO
# ---------------------------------------------------------------------------
RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
 && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
 && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
 && ln -sf /usr/local/bin/pip3 /usr/local/bin/pip

# ---------------------------------------------------------------------------
# uv
# ---------------------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1

# ---------------------------------------------------------------------------
# 1. PyTorch con CUDA 11.8 (combinación probada en local)
# ---------------------------------------------------------------------------
RUN uv pip install --system torch==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu118

# ---------------------------------------------------------------------------
# 2. mamba-ssm y causal-conv1d (requieren compilación con nvcc)
#    - packaging, ninja, setuptools, wheel son build-deps de setup.py
#    - ninja paraleliza la compilación de kernels CUDA (~3x más rápido)
#    - --no-build-isolation es OBLIGATORIO: el setup.py de causal-conv1d
#      hace `import torch`, y con build isolation pip crea un env limpio
#      sin torch y falla con ModuleNotFoundError: No module named 'torch'.
#    - TORCH_CUDA_ARCH_LIST evita detectar GPUs en build (no hay GPU en Cloud Build)
#      y compila PTX para las arquitecturas que usarás en Vertex (T4/V100/A100/L4/H100).
# ---------------------------------------------------------------------------
RUN python3.11 -m pip install --no-cache-dir packaging ninja setuptools wheel

ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" \
    MAMBA_FORCE_BUILD=TRUE \
    CAUSAL_CONV1D_FORCE_BUILD=TRUE

# NOTA: instalamos desde GitHub (no desde PyPI) porque el sdist de
# mamba-ssm==2.2.2 en PyPI está incompleto: le faltan archivos bajo
# csrc/selective_scan/ y ninja falla con:
#   "ninja: error: '.../csrc/selective_scan/selective_scan.cpp', missing"
# El tag v2.2.2 del repo sí trae el árbol completo.
RUN python3.11 -m pip install --no-cache-dir --no-build-isolation \
    git+https://github.com/Dao-AILab/causal-conv1d.git@v1.4.0 \
    git+https://github.com/state-spaces/mamba.git@v2.2.2

# ---------------------------------------------------------------------------
# 3. Resto de dependencias
# ---------------------------------------------------------------------------
RUN uv pip install --system \
    tzdata \
    "transformers>=4.35,<5.0" \
    polars==1.39.3 \
    numpy==2.2.6 \
    scikit-learn==1.7.2 \
    pyarrow==23.0.1 \
    wandb==0.25.1 \
    optuna==4.8.0 \
    gcsfs==2026.1.0 \
    fsspec==2026.1.0 \
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
# Experimento: bakeamos el .env en la imagen por simplicidad.
# NO hacer esto en producción — los secretos quedan en las capas de la imagen.
COPY .env .

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
CMD ["python", "Mamba.py"]
