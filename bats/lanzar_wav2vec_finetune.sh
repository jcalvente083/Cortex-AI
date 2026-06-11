#!/usr/bin/env bash
# lanzar_wav2vec_finetune.sh — Wav2Vec2 finetune en servidor Linux
# Uso: bash bats/lanzar_wav2vec_finetune.sh
# Log: logs/finetune_<timestamp>.log  (además de stdout en tiempo real)
# Tiempo estimado: 12-38 h por run según GPU

set -euo pipefail

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/finetune_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

# Redirige stdout+stderr al log y también a la terminal
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo " Cortex-AI -- Wav2Vec2 Finetune runs"
echo " Inicio: $(date)"
echo " Config: freeze layers 0-20, batch 4, grad_accum 4 (batch efectivo 16)"
echo " Tiempo estimado: 12-38 h por run"
echo " Log: ${LOG_FILE}"
echo "============================================================"
echo ""

# ---------- Comprobaciones previas ----------
if ! command -v uv &>/dev/null; then
    echo "[!] 'uv' no encontrado. Instálalo con: curl -Lsf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

if uv run python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU=$(uv run python -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "[✓] GPU detectada: ${GPU}"
else
    echo "[!] AVISO: no se detecta CUDA. El finetune sin GPU tardará días."
    read -r -p "    ¿Continuar de todas formas? [s/N] " resp
    [[ "$resp" =~ ^[sS]$ ]] || exit 1
fi
echo ""

# ---------- Run 1: baseline ----------
echo "[1/2] Wav2Vec2 Finetune -- baseline (con edad, todos los datos)"
echo "Inicio: $(date)"
uv run python -m src.models.train_wav2vec_finetune --run baseline --epochs 30
echo "Fin:   $(date)"
echo ""

# ---------- Run 2: age_matched ----------
echo "[2/2] Wav2Vec2 Finetune -- age_matched (sin confound de edad)"
echo "Inicio: $(date)"
uv run python -m src.models.train_wav2vec_finetune --run age_matched --age-match --epochs 30
echo "Fin:   $(date)"
echo ""

echo "============================================================"
echo " Todos los runs completados: $(date)"
echo " Modelos  -> models/wav2vec_finetune/"
echo " Figuras  -> reports/wav2vec_finetune/"
echo " Log      -> ${LOG_FILE}"
echo "============================================================"
