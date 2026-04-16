#!/bin/bash
# =============================================================================
# submit_experiments.sh — Lanza múltiples experimentos en Vertex AI en paralelo
# =============================================================================
set -e

PROJECT_ID="composed-arch-276322"
REGION="us-central1"

# Combinaciones de experimentos a correr
# Formato: "EXP,USAR_BALANCE,TASK"
EXPERIMENTS=(
    "1,False,next_action"
    "2,True,next_action"
    "3,False,both"
    "4,True,both"
)

echo "=========================================================="
echo "  Enviando ${#EXPERIMENTS[@]} experimentos a Vertex AI..."
echo "=========================================================="

for exp_config in "${EXPERIMENTS[@]}"; do
    # Extraer variables separadas por coma
    IFS=',' read -r EXP USAR_BALANCE TASK <<< "$exp_config"
    
    JOB_DISPLAY_NAME="mamba-exp${EXP}-${TASK}-bal${USAR_BALANCE}"
    
    echo "▶ Lanzando: $JOB_DISPLAY_NAME"
    
    # Lanza el job en background y sobrescribe las variables de entorno
    gcloud ai custom-jobs create \
        --region="${REGION}" \
        --display-name="${JOB_DISPLAY_NAME}" \
        --config=vertex_job.yaml \
        --args="--set-env-vars=EXP=${EXP},USAR_BALANCE=${USAR_BALANCE},TASK=${TASK}" \
        --project="${PROJECT_ID}" &
        
    # sleep 2s para no saturar la API
    sleep 2
done

# Esperar a que terminen los comandos de gcloud (no el entrenamiento en sí)
wait

echo ""
echo "✅ Todos los jobs fueron enviados."
echo "   Monitorea en: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
