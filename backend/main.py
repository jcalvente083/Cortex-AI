"""
main.py — Cortex-AI Backend (FastAPI).

Endpoints:
  GET  /health              Estado del servidor y modelos disponibles
  GET  /modelos             Lista de modelos y capacidades
  POST /predict             Inferencia con un solo audio
  POST /predict/batch       Inferencia con 3 audios (vocal + frase + espontanea)

Modelos locales (RPi5):  knn | xgboost | resnet18
Modelos cloud (Modal):   wav2vec_embeddings | wav2vec_finetune

Arranque:
  uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
from contextlib import asynccontextmanager

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.config import ACTIVITIES, ALL_MODELS, CLOUD_API_URL, CLOUD_MODELS, LOCAL_MODELS
from backend.inference.cloud import predict_cloud
from backend.inference.local import (
    load_local_models,
    predict_knn,
    predict_resnet18,
    predict_xgboost,
    compute_grad_cam_from_bytes,
)
from backend.schemas import (
    BatchPrediccionResponse,
    Explicabilidad,
    HealthResponse,
    ModeloInfo,
    PrediccionResponse,
)

# =============================================================================
# LIFESPAN — carga modelos al arrancar
# =============================================================================
_registry: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _registry
    print("[startup] Cargando modelos locales...")
    _registry = load_local_models()
    print("[startup] API lista.\n")
    yield
    print("[shutdown] Cerrando API.")


# =============================================================================
# APP
# =============================================================================
app = FastAPI(
    title="Cortex-AI API",
    description="Detección de Parkinson mediante análisis de voz",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# HELPERS
# =============================================================================

def _nivel_riesgo(prob: float, umbral: float) -> str:
    if prob >= umbral:
        return "Alto"
    if prob >= umbral * 0.6:
        return "Medio"
    return "Bajo"


def _modelo_ok(modelo: str, actividad: str) -> bool:
    if modelo in LOCAL_MODELS:
        return bool(_registry.get(modelo, {}).get(actividad))
    return bool(CLOUD_API_URL)


async def _inferir(audio_bytes: bytes, modelo: str, actividad: str):
    """Despacha la inferencia al proveedor correcto y devuelve (prob, umbral, expl)."""
    if modelo == "knn":
        return predict_knn(audio_bytes, actividad, _registry)
    if modelo == "xgboost":
        return predict_xgboost(audio_bytes, actividad, _registry)
    if modelo == "resnet18":
        return predict_resnet18(audio_bytes, actividad, _registry)

    # Modelos cloud
    result = await predict_cloud(audio_bytes, modelo, actividad)
    prob   = result["probabilidad_pd"]
    umbral = result["umbral"]
    expl   = Explicabilidad(**result.get("explicabilidad", {"disponible": False}))
    return prob, umbral, expl


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    _meta = {
        "knn": (
            "KNN sobre features acústicas (ShimmerDb, ATRI, Hnr, CHNR, rPPQ)",
            True, "local",
        ),
        "xgboost": (
            "XGBoost sobre features acústicas con SHAP",
            True, "local",
        ),
        "resnet18": (
            "ResNet18 fine-tuned sobre mel-espectrogramas",
            False, "local",
        ),
        "wav2vec_embeddings": (
            "Wav2Vec2 (embeddings congelados) + KNN",
            False, "cloud",
        ),
        "wav2vec_finetune": (
            "Wav2Vec2 fine-tuning end-to-end (audio crudo)",
            False, "cloud",
        ),
    }

    modelos = []
    for nombre, (desc, expl, ubicacion) in _meta.items():
        if nombre in LOCAL_MODELS:
            acts = [a for a in ACTIVITIES if _registry.get(nombre, {}).get(a)]
        else:
            acts = ACTIVITIES if CLOUD_API_URL else []

        modelos.append(ModeloInfo(
            nombre=nombre,
            disponible=bool(acts),
            actividades=acts,
            explicabilidad=expl,
            ubicacion=ubicacion,
            descripcion=desc,
        ))

    return HealthResponse(
        status="ok",
        cloud_disponible=bool(CLOUD_API_URL),
        modelos=modelos,
    )


@app.get("/modelos")
async def get_modelos():
    h = await health()
    return {"modelos": [m.model_dump() for m in h.modelos]}


@app.post("/predict", response_model=PrediccionResponse)
async def predict(
    audio:    UploadFile = File(..., description="Archivo de audio (wav, mp3, m4a…)"),
    modelo:   str        = Form(..., description="knn | xgboost | resnet18 | wav2vec_embeddings | wav2vec_finetune"),
    actividad: str       = Form(..., description="vocal | frase | espontanea | all"),
):
    if modelo not in ALL_MODELS:
        raise HTTPException(400, f"Modelo desconocido: '{modelo}'. Opciones: {sorted(ALL_MODELS)}")
    if actividad not in ACTIVITIES:
        raise HTTPException(400, f"Actividad desconocida: '{actividad}'. Opciones: {ACTIVITIES}")
    if not _modelo_ok(modelo, actividad):
        raise HTTPException(
            503,
            f"Modelo '{modelo}' no disponible para actividad '{actividad}'. "
            "Para modelos Wav2Vec2 configura CLOUD_API_URL.",
        )

    audio_bytes = await audio.read()

    try:
        prob_pd, umbral, expl = await _inferir(audio_bytes, modelo, actividad)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(500, f"Error interno: {e}")

    return PrediccionResponse(
        probabilidad_pd=round(prob_pd, 4),
        prediccion="Parkinson" if prob_pd >= umbral else "Sano",
        nivel_riesgo=_nivel_riesgo(prob_pd, umbral),
        umbral=round(umbral, 4),
        modelo=modelo,
        actividad=actividad,
        explicabilidad=expl,
    )


@app.post("/predict/batch", response_model=BatchPrediccionResponse)
async def predict_batch(
    audio_vocal:      UploadFile = File(..., description="Audio de vocal sostenida"),
    audio_frase:      UploadFile = File(..., description="Audio de frase leída"),
    audio_espontanea: UploadFile = File(..., description="Audio de habla espontánea"),
    modelo:           str        = Form(..., description="knn | xgboost | resnet18 | wav2vec_*"),
):
    """
    Recibe los 3 audios de una sesión completa, usa el modelo 'all' para cada uno
    y promedia las probabilidades → resultado final de la sesión.
    """
    if modelo not in ALL_MODELS:
        raise HTTPException(400, f"Modelo desconocido: '{modelo}'")
    if not _modelo_ok(modelo, "all"):
        raise HTTPException(503, f"Modelo '{modelo}/all' no disponible")

    audios = {
        "vocal":      await audio_vocal.read(),
        "frase":      await audio_frase.read(),
        "espontanea": await audio_espontanea.read(),
    }

    probs, umbrales, expls = {}, {}, {}
    for actividad, audio_bytes in audios.items():
        try:
            prob, umbral, expl = await _inferir(audio_bytes, modelo, "all")
            probs[actividad]    = round(prob, 4)
            umbrales[actividad] = round(umbral, 4)
            expls[actividad]    = expl
        except Exception as e:
            raise HTTPException(422, f"Error procesando audio '{actividad}': {e}")

    prob_final   = round(sum(probs.values()) / len(probs), 4)
    umbral_medio = round(sum(umbrales.values()) / len(umbrales), 4)

    # Explicabilidad: del audio con mayor probabilidad (más informativo)
    best_act   = max(probs, key=lambda k: probs[k])
    mejor_expl = expls[best_act]

    # Grad-CAM: solo para ResNet18
    grad_cam = None
    if modelo == "resnet18":
        resnet_data = _registry.get("resnet18", {}).get("all")
        if resnet_data:
            grad_cam = {
                act: compute_grad_cam_from_bytes(ab, resnet_data["model"])
                for act, ab in audios.items()
            }

    return BatchPrediccionResponse(
        probabilidad_pd_final=prob_final,
        prediccion="Parkinson" if prob_final >= umbral_medio else "Sano",
        nivel_riesgo=_nivel_riesgo(prob_final, umbral_medio),
        umbral_promedio=umbral_medio,
        modelo=modelo,
        detalle_por_actividad=probs,
        explicabilidad=mejor_expl,
        grad_cam_por_actividad=grad_cam,
    )
