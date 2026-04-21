# =============================================================================
# Dockerfile — imagen app (solo código fuente)
# Base: gcr.io/PROJECT_ID/mamba-base:v1 (dependencias compiladas)
# Para reconstruir la base: bash deploy_base.sh
# =============================================================================
FROM gcr.io/composed-arch-276322/mamba-base:v1

WORKDIR /app
COPY Mamba.py .
COPY model_arch.py .
COPY eval_contrastive_clusters.py .
COPY split_estratificado_uids.json .
COPY .env .

CMD ["python", "Mamba.py"]
