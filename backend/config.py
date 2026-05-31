"""
config.py — Rutas de modelos y variables de entorno del backend.
"""

import os
from pathlib import Path

# ── Raíz del proyecto (un nivel arriba de backend/) ──────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent

# ── Rutas de modelos entrenados ───────────────────────────────────────────────
MODELS_KNN    = PROJECT_ROOT / "models/traditionals/KNN/sin_age"
MODELS_XGB    = PROJECT_ROOT / "models/traditionals/XGBoost/sin_age"
MODELS_RESNET = PROJECT_ROOT / "models/resnet/ResNet18/specaugment_v2"

# ── Cloud ─────────────────────────────────────────────────────────────────────
CLOUD_API_URL = os.getenv("CLOUD_API_URL", "")

# ── Audio ─────────────────────────────────────────────────────────────────────
SR     = 16000
TOP_DB = 20

# ── Clasificación de modelos ──────────────────────────────────────────────────
LOCAL_MODELS = {"knn", "xgboost", "resnet18"}
CLOUD_MODELS = {"wav2vec_embeddings", "wav2vec_finetune"}
ALL_MODELS   = LOCAL_MODELS | CLOUD_MODELS
ACTIVITIES   = ["vocal", "frase", "espontanea", "all"]
