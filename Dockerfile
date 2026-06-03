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
COPY postgres_apipythondb_producto.csv .
COPY .env .

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

CMD ["./entrypoint.sh"]
