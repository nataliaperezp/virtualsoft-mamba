#!/bin/bash
# =============================================================================
# deploy_sessions.sh — Build de imagen + lanzamiento de job Mamba-Sessions
# Uso:
#   bash deploy_sessions.sh                          # build + lanzamiento interactivo
#   bash deploy_sessions.sh --skip-build             # solo lanzar (imagen ya existe)
#   bash deploy_sessions.sh --run 1,3 --machine g16  # build + lanzar exp 1 y 3
#   bash deploy_sessions.sh --dry-run --run 2        # build + dry-run (no crea job)
#   bash deploy_sessions.sh --test --run 1           # build + prueba rápida
# =============================================================================
set -e

PROJECT_ID="composed-arch-276322"
IMAGE="gcr.io/${PROJECT_ID}/mamba-sessions:v1"

# Cambiar al directorio raíz del proyecto (para que env, json, csv se incluyan en el build context)
cd "$(dirname "$0")/.."

SKIP_BUILD=false
LAUNCH_ARGS=()

# Parsear flags propios de este script
for arg in "$@"; do
    case $arg in
        --skip-build) SKIP_BUILD=true ;;
        *)            LAUNCH_ARGS+=("$arg") ;;
    esac
done

# ------------------------------------------------------------------------------
# PASO 1: Build y Push de la imagen con Cloud Build
# ------------------------------------------------------------------------------
if [ "$SKIP_BUILD" = false ]; then
    echo ""
    echo "▶ [1/2] Construyendo imagen ${IMAGE} con Cloud Build (~30-40 min)..."
    gcloud builds submit \
        --config=Mamba_sessions/cloudbuild.sessions.yaml \
        --project="${PROJECT_ID}" \
        .
    echo "✅ Imagen publicada: ${IMAGE}"
else
    echo "⏩ [1/2] Skipping build — usando imagen existente: ${IMAGE}"
fi

# ------------------------------------------------------------------------------
# PASO 2: Lanzar experimento(s) en Vertex AI
# ------------------------------------------------------------------------------
echo ""
echo "▶ [2/2] Lanzando job en Vertex AI..."
python3 Mamba_sessions/launch_sessions.py "${LAUNCH_ARGS[@]}"
