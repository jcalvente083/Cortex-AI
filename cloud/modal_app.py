"""
modal_app.py — Cortex-AI Cloud API (Modal.com)

Sirve los modelos Wav2Vec2 que no caben en la RPi5.
Los pesos se almacenan en un Modal Volume que se crea una sola vez.

── SETUP (ejecutar una sola vez) ──────────────────────────────────────────────
  pip install modal
  modal setup                          # login con GitHub/Google
  modal volume create cortex-models   # crear el volumen persistente

── SUBIR LOS MODELOS AL VOLUMEN ───────────────────────────────────────────────
  # Wav2Vec2 embeddings + KNN
  modal volume put cortex-models models/wav2vec/ /models/wav2vec/

  # Wav2Vec2 fine-tuning
  modal volume put cortex-models models/wav2vec_finetune/ /models/wav2vec_finetune/

── DEPLOY ──────────────────────────────────────────────────────────────────────
  modal deploy cloud/modal_app.py

  → La URL estable aparece en consola:
    https://<tu-usuario>--cortex-ai-predict.modal.run

── CONFIGURAR EN RPI5 ──────────────────────────────────────────────────────────
  export CLOUD_API_URL=https://<tu-usuario>--cortex-ai-predict.modal.run

── PRUEBA RÁPIDA ───────────────────────────────────────────────────────────────
  curl -X GET https://<tu-usuario>--cortex-ai-health.modal.run
"""

import io
import warnings

import modal

warnings.filterwarnings("ignore")

# =============================================================================
# IMAGEN — dependencias del contenedor
# =============================================================================
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi",
        "python-multipart",
        "transformers",
        "torch",
        "torchaudio",
        "librosa",
        "numpy",
        "joblib",
        "scikit-learn",
        "timm",
    )
)

# =============================================================================
# APP + VOLUMEN
# =============================================================================
app     = modal.App("cortex-ai", image=image)
volume  = modal.Volume.from_name("cortex-models")
MOUNT   = "/models"   # punto de montaje dentro del contenedor

SR            = 16_000
MAX_AUDIO_SECS = 10
MAX_SAMPLES   = SR * MAX_AUDIO_SECS


# =============================================================================
# ESTADO GLOBAL (cargado una vez por contenedor caliente)
# =============================================================================
class _State:
    feature_extractor = None
    models: dict      = {}          # {(modelo, actividad): artefactos}


_state = _State()


def _load_wav2vec_extractor():
    from transformers import Wav2Vec2FeatureExtractor
    model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
    return Wav2Vec2FeatureExtractor.from_pretrained(model_name)


def _load_wav2vec_model_finetune(folder: str):
    """Carga el Wav2Vec2Classifier fine-tuneado desde el volumen."""
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Model
    from pathlib import Path
    import joblib

    class Wav2Vec2Classifier(nn.Module):
        def __init__(self, model_name, dropout=0.1):
            super().__init__()
            self.backbone   = Wav2Vec2Model.from_pretrained(model_name)
            hidden_size     = self.backbone.config.hidden_size
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 2),
            )

        def forward(self, x):
            out    = self.backbone(x)
            pooled = out.last_hidden_state.mean(dim=1)
            return self.classifier(pooled)

    model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
    model      = Wav2Vec2Classifier(model_name)
    path       = Path(folder) / "model.pth"
    state      = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    threshold = joblib.load(Path(folder) / "threshold.pkl")
    return {"model": model, "threshold": threshold, "type": "finetune"}


def _load_wav2vec_embeddings_knn(folder: str):
    """Carga el pipeline embeddings+KNN desde el volumen."""
    import joblib
    from pathlib import Path
    return {
        "model":     joblib.load(Path(folder) / "model.pkl"),
        "scaler":    joblib.load(Path(folder) / "scaler.pkl"),
        "pca":       joblib.load(Path(folder) / "pca.pkl"),
        "threshold": joblib.load(Path(folder) / "threshold.pkl"),
        "emb_cols":  joblib.load(Path(folder) / "emb_cols.pkl"),
        "type":      "embeddings",
    }


def _ensure_loaded(modelo: str, actividad: str):
    """Carga lazy del modelo pedido. Solo una vez por contenedor."""
    key = (modelo, actividad)
    if key in _state.models:
        return

    if modelo == "wav2vec_finetune":
        folder = f"{MOUNT}/wav2vec_finetune/Wav2Vec2/baseline/{actividad}"
        _state.models[key] = _load_wav2vec_model_finetune(folder)

    elif modelo == "wav2vec_embeddings":
        folder = f"{MOUNT}/wav2vec/KNN/baseline/{actividad}"
        _state.models[key] = _load_wav2vec_embeddings_knn(folder)

    if _state.feature_extractor is None:
        _state.feature_extractor = _load_wav2vec_extractor()


# =============================================================================
# INFERENCIA
# =============================================================================

def _preprocess_audio(audio_bytes: bytes) -> "np.ndarray":
    import numpy as np
    import librosa
    y, _ = librosa.load(io.BytesIO(audio_bytes), sr=SR, mono=True)
    if len(y) >= MAX_SAMPLES:
        y = y[:MAX_SAMPLES]
    else:
        y = np.pad(y, (0, MAX_SAMPLES - len(y)))
    return y.astype(np.float32)


def _infer_finetune(audio_bytes: bytes, actividad: str) -> tuple[float, float]:
    import torch
    import torch.nn.functional as F

    _ensure_loaded("wav2vec_finetune", actividad)
    data = _state.models[("wav2vec_finetune", actividad)]

    y      = _preprocess_audio(audio_bytes)
    inputs = _state.feature_extractor(
        y, sampling_rate=SR, return_tensors="pt", padding=False,
    )
    input_values = inputs.input_values   # (1, MAX_SAMPLES)

    with torch.no_grad():
        logits = data["model"](input_values)
        prob   = float(F.softmax(logits, dim=1)[0, 1].item())

    return prob, float(data["threshold"])


def _infer_embeddings(audio_bytes: bytes, actividad: str) -> tuple[float, float]:
    import torch
    import numpy as np
    from transformers import Wav2Vec2Model

    _ensure_loaded("wav2vec_embeddings", actividad)
    data = _state.models[("wav2vec_embeddings", actividad)]

    # Extraer embedding con Wav2Vec2 congelado
    if not hasattr(_infer_embeddings, "_backbone"):
        model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
        _infer_embeddings._backbone = Wav2Vec2Model.from_pretrained(model_name).eval()

    y      = _preprocess_audio(audio_bytes)
    inputs = _state.feature_extractor(
        y, sampling_rate=SR, return_tensors="pt", padding=False,
    )
    with torch.no_grad():
        hidden = _infer_embeddings._backbone(inputs.input_values).last_hidden_state
        emb    = hidden.mean(dim=1).squeeze().numpy()   # (1024,)

    emb_cols = data["emb_cols"]
    X        = emb.reshape(1, -1)[:, :len(emb_cols)]
    X_sc     = data["scaler"].transform(X)
    X_pca    = data["pca"].transform(X_sc)
    prob     = float(data["model"].predict_proba(X_pca)[0, 1])
    return prob, float(data["threshold"])


# =============================================================================
# ENDPOINT FASTAPI — expuesto en Modal como web endpoint
# =============================================================================

@app.function(
    gpu="T4",
    volumes={MOUNT: volume},
    timeout=180,
    container_idle_timeout=300,   # contenedor caliente 5 min tras la última petición
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, File, Form, UploadFile, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    api = FastAPI(title="Cortex-AI Cloud", version="1.0.0")
    api.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    ACTIVITIES = ["vocal", "frase", "espontanea", "all"]

    @api.get("/health")
    async def health():
        return {"status": "ok", "device": "cuda", "modelos": ["wav2vec_finetune", "wav2vec_embeddings"]}

    @api.post("/predict")
    async def predict(
        audio:    UploadFile = File(...),
        modelo:   str        = Form(...),
        actividad: str       = Form(...),
    ):
        if modelo not in {"wav2vec_finetune", "wav2vec_embeddings"}:
            raise HTTPException(400, f"Modelo '{modelo}' no soportado en cloud")
        if actividad not in ACTIVITIES:
            raise HTTPException(400, f"Actividad '{actividad}' no reconocida")

        audio_bytes = await audio.read()

        try:
            if modelo == "wav2vec_finetune":
                prob, umbral = _infer_finetune(audio_bytes, actividad)
            else:
                prob, umbral = _infer_embeddings(audio_bytes, actividad)
        except FileNotFoundError:
            raise HTTPException(
                503,
                f"Modelo '{modelo}/{actividad}' no encontrado en el volumen. "
                "Sube los pesos con: modal volume put cortex-models ...",
            )
        except Exception as e:
            raise HTTPException(500, f"Error de inferencia: {e}")

        return {
            "probabilidad_pd": round(prob, 4),
            "prediccion":      "Parkinson" if prob >= umbral else "Control",
            "nivel_riesgo":    "Alto" if prob >= umbral else ("Medio" if prob >= umbral * 0.6 else "Bajo"),
            "umbral":          round(umbral, 4),
            "modelo":          modelo,
            "actividad":       actividad,
            "explicabilidad":  {"disponible": False},
        }

    return api
