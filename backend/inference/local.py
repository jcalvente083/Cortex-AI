"""
local.py — Inferencia con modelos locales: KNN, XGBoost, ResNet18.

Todos los modelos se cargan en memoria al arrancar la API (load_local_models).
"""

import warnings
import joblib
from pathlib import Path

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import shap

warnings.filterwarnings("ignore")

from backend.config import MODELS_KNN, MODELS_XGB, MODELS_RESNET, SR
from backend.inference.extractor import preprocess_waveform, extract_acoustic_features
from backend.schemas import FeatureInfo, Explicabilidad

# ── Parámetros ResNet18 (idénticos a train_resnet.py) ────────────────────────
FRAME_LEN_S = 0.4
HOP_LEN_S   = 0.2
N_MELS      = 65
N_FFT       = 512
N_FFT_HOP_S = 0.03
RESIZE      = 224

ACTIVITIES = ["vocal", "frase", "espontanea", "all"]


# =============================================================================
# CARGA DE MODELOS
# =============================================================================

def _load_traditional(folder: Path) -> dict | None:
    """Carga model.pkl + scaler.pkl + threshold.pkl + features.pkl."""
    try:
        return {
            "model":     joblib.load(folder / "model.pkl"),
            "scaler":    joblib.load(folder / "scaler.pkl"),
            "threshold": joblib.load(folder / "threshold.pkl"),
            "features":  joblib.load(folder / "features.pkl"),
        }
    except Exception:
        return None


def _build_resnet18() -> nn.Module:
    model = timm.create_model("resnet18", pretrained=False, num_classes=2)
    model.conv1 = nn.Conv2d(
        1, model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False,
    )
    return model


def _load_resnet(folder: Path) -> dict | None:
    try:
        model = _build_resnet18()
        state = torch.load(folder / "model.pth", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return {
            "model":     model,
            "threshold": joblib.load(folder / "threshold.pkl"),
        }
    except Exception:
        return None


def load_local_models() -> dict:
    """
    Carga KNN, XGBoost y ResNet18 para todas las actividades.
    Llamar una sola vez al arrancar la API (lifespan).
    """
    registry: dict = {"knn": {}, "xgboost": {}, "resnet18": {}}

    for act in ACTIVITIES:
        if (d := _load_traditional(MODELS_KNN / act)):
            registry["knn"][act] = d

        if (d := _load_traditional(MODELS_XGB / act)):
            registry["xgboost"][act] = d

        if (d := _load_resnet(MODELS_RESNET / act)):
            registry["resnet18"][act] = d

    loaded = {m: list(a.keys()) for m, a in registry.items() if a}
    print(f"[models] Cargados: {loaded}")
    return registry


# =============================================================================
# KNN
# =============================================================================

def predict_knn(
    audio_bytes: bytes,
    activity:    str,
    registry:    dict,
) -> tuple[float, float, Explicabilidad]:
    """→ (prob_pd, umbral, explicabilidad)"""
    data = registry["knn"].get(activity)
    if data is None:
        raise ValueError(f"KNN: sin modelo para actividad '{activity}'")

    features: list[str] = data["features"]
    feats = extract_acoustic_features(audio_bytes, features)
    if not feats:
        raise ValueError("No se pudieron extraer features acústicas del audio")

    none_feats = [f for f in features if feats.get(f) is None]
    if none_feats:
        print(f"[knn/{activity}] Features None (imputando 0.0): {none_feats}")

    X    = np.array([[feats.get(f) if feats.get(f) is not None else 0.0 for f in features]])
    X_sc = data["scaler"].transform(X)
    prob_pd = float(data["model"].predict_proba(X_sc)[0, 1])
    umbral  = float(data["threshold"])

    # Explicabilidad: z-score de cada feature normalizado a [-1, 1]
    feat_infos = []
    for i, fname in enumerate(features):
        z     = float(X_sc[0, i])
        contr = float(np.tanh(z))
        feat_infos.append(FeatureInfo(
            nombre=fname,
            valor=round(float(feats[fname]), 4),
            contribucion=round(contr, 4),
            direccion="positivo" if contr > 0.05 else ("negativo" if contr < -0.05 else "neutro"),
        ))

    expl = Explicabilidad(
        disponible=True,
        tipo="feature_deviation",
        features=sorted(feat_infos, key=lambda x: abs(x.contribucion), reverse=True),
    )
    return prob_pd, umbral, expl


# =============================================================================
# XGBOOST + SHAP
# =============================================================================

def predict_xgboost(
    audio_bytes: bytes,
    activity:    str,
    registry:    dict,
) -> tuple[float, float, Explicabilidad]:
    data = registry["xgboost"].get(activity)
    if data is None:
        raise ValueError(f"XGBoost: sin modelo para actividad '{activity}'")

    features: list[str] = data["features"]
    feats = extract_acoustic_features(audio_bytes, features)
    if not feats:
        raise ValueError("No se pudieron extraer features acústicas del audio")

    none_feats = [f for f in features if feats.get(f) is None]
    if none_feats:
        print(f"[xgb/{activity}] Features None (imputando 0.0): {none_feats}")

    X    = np.array([[feats.get(f) if feats.get(f) is not None else 0.0 for f in features]])
    X_sc = data["scaler"].transform(X)
    prob_pd = float(data["model"].predict_proba(X_sc)[0, 1])
    umbral  = float(data["threshold"])

    # SHAP via TreeExplainer
    try:
        explainer = shap.TreeExplainer(data["model"])
        shap_exp  = explainer(X_sc)           # nueva API: Explanation object

        # shap_exp.values: (n_samples, n_features, n_classes) o (n_samples, n_features)
        sv = shap_exp.values[0]
        if sv.ndim == 2:                      # (n_features, n_classes) → tomar clase PD
            sv = sv[:, 1]
        bv = shap_exp.base_values[0]
        base_value = float(bv[1] if hasattr(bv, "__len__") else bv)

        feat_infos = []
        for i, fname in enumerate(features):
            s = float(sv[i])
            feat_infos.append(FeatureInfo(
                nombre=fname,
                valor=round(float(feats[fname]), 4),
                contribucion=round(s, 4),
                direccion="positivo" if s > 0.01 else ("negativo" if s < -0.01 else "neutro"),
            ))

        expl = Explicabilidad(
            disponible=True,
            tipo="shap",
            features=sorted(feat_infos, key=lambda x: abs(x.contribucion), reverse=True),
            base_value=round(base_value, 4),
        )
    except Exception as e:
        print(f"[shap] {e}")
        expl = Explicabilidad(disponible=False)

    return prob_pd, umbral, expl


# =============================================================================
# RESNET18
# =============================================================================

def _audio_to_frames(y: np.ndarray) -> torch.Tensor:
    """Waveform → tensor de frames mel-espectrogramas (224×224, 1 canal)."""
    from torchvision.transforms.functional import resize

    frame_len = int(SR * FRAME_LEN_S)
    hop_len   = int(SR * HOP_LEN_S)
    n_fft_hop = int(SR * N_FFT_HOP_S)

    if len(y) >= frame_len:
        y_framed = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_len)
    else:
        y_pad    = np.concatenate([y, np.zeros(frame_len - len(y))])
        y_framed = librosa.util.frame(y_pad, frame_length=frame_len, hop_length=hop_len)

    mels    = librosa.feature.melspectrogram(
        y=y_framed.T, sr=SR, n_fft=N_FFT, hop_length=n_fft_hop, n_mels=N_MELS,
    )
    mels_db   = librosa.power_to_db(mels, ref=np.max)
    mels_norm = (mels_db - mels_db.mean()) / (mels_db.std() + 1e-8)

    tensors = []
    for i in range(mels_norm.shape[0]):
        t = torch.from_numpy(mels_norm[i]).float().unsqueeze(0)
        t = resize(t, [RESIZE, RESIZE], antialias=True)
        tensors.append(t)
    return torch.stack(tensors)   # (n_frames, 1, 224, 224)


def predict_resnet18(
    audio_bytes: bytes,
    activity:    str,
    registry:    dict,
) -> tuple[float, float, Explicabilidad]:
    data = registry["resnet18"].get(activity)
    if data is None:
        raise ValueError(f"ResNet18: sin modelo para actividad '{activity}'")

    y      = preprocess_waveform(audio_bytes)
    frames = _audio_to_frames(y)   # (n_frames, 1, 224, 224)

    with torch.no_grad():
        logits  = data["model"](frames)
        probs   = F.softmax(logits, dim=1)[:, 1]

    prob_pd = float(probs.mean().item())
    umbral  = float(data["threshold"])

    return prob_pd, umbral, Explicabilidad(disponible=False)
