#!/bin/bash
# =============================================================================
# entrypoint_sessions.sh — Lanzador para Mamba_Sessions.py
# Single-GPU: python normal | Multi-GPU: torchrun (DDP)
# =============================================================================

NUM_GPUS=${NUM_GPUS:-1}

echo "🔧 GPUs detectadas: ${NUM_GPUS}"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "🚀 Lanzando Mamba_Sessions con torchrun (DDP) — ${NUM_GPUS} procesos"
    exec torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --master_addr=localhost \
        --master_port=29500 \
        Mamba_Sessions.py
else
    echo "🚀 Lanzando Mamba_Sessions con python (single GPU)"
    exec python Mamba_Sessions.py
fi
