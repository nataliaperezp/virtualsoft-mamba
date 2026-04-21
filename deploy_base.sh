#!/bin/bash
PROJECT_ID="composed-arch-276322"

echo "▶ Construyendo imagen base con dependencias CUDA (~30-40 min, solo cuando cambien deps)..."
gcloud builds submit \
    --config=cloudbuild.base.yaml \
    --project="${PROJECT_ID}" \
    .

echo "✅ Base publicada en: gcr.io/${PROJECT_ID}/mamba-base:v1"
