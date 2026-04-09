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

# Alias python
RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python

# ---------------------------------------------------------------------------
# uv
# ---------------------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1

# ---------------------------------------------------------------------------
# Dependencias
# ---------------------------------------------------------------------------
WORKDIR /app

COPY pyproject.toml .

# uv instala todo en el Python del sistema (UV_SYSTEM_PYTHON=1)
# --no-dev evita instalar dependencias de desarrollo si las hubiera
RUN uv pip install --system -r pyproject.toml

# ---------------------------------------------------------------------------
# Código fuente
# ---------------------------------------------------------------------------
COPY Mamba.py .

# El JSON de UIDs se copia si está disponible localmente;
# si prefieres leerlo desde GCS, ajusta UIDS_PATH en el Job spec.
COPY split_estratificado_uids.json .

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
CMD ["python", "Mamba.py"]
