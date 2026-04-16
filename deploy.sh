# #!/bin/bash
# # =============================================================================
# # deploy.sh — Script de despliegue completo para Vertex AI Training Job
# # Uso: bash deploy.sh
# # =============================================================================
# set -e  # Salir si cualquier comando falla

PROJECT_ID="composed-arch-276322"
REGION="us-central1"
IMAGE="gcr.io/${PROJECT_ID}/mamba-trainer-contrastive:v1"        

# echo "=============================================="
# echo "  Desplegando Mamba Training Job en Vertex AI"
# echo "=============================================="

# # ------------------------------------------------------------------------------
# # PASO 1: Autenticación
# # ------------------------------------------------------------------------------
# echo ""
# echo "▶ [1/4] Autenticando en GCP..."
# gcloud config set project "${PROJECT_ID}"
# gcloud auth configure-docker --quiet

# ------------------------------------------------------------------------------
# PASO 3: Build y Push de la imagen con Cloud Build
# (Se construye en la nube — evita problemas de ARM en Mac)
# ------------------------------------------------------------------------------
echo ""
echo "▶ [3/4] Construyendo imagen con Cloud Build (puede tardar ~30-40 min)..."
gcloud builds submit \
    --config=cloudbuild.yaml \
    --project="${PROJECT_ID}" \
    .

echo "✅ Imagen publicada en: ${IMAGE}"

# ------------------------------------------------------------------------------
# PASO 4: Lanzar el Custom Training Job en Vertex AI
# ------------------------------------------------------------------------------
#echo ""
#echo "▶ [4/4] Lanzando Vertex AI Training Job..."
#gcloud ai custom-jobs create \
#    --region="${REGION}" \
#    --display-name="mamba-training-$(date +%Y%m%d-%H%M)" \
#    --config=vertex_job.yaml \
#    --project="${PROJECT_ID}"

#echo ""
#echo "✅ Job enviado. Monitorea en:"
#echo "   https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
