"""
extractor.py — Preprocesamiento de audio y extracción de features acústicas.

Replica exactamente el pipeline de entrenamiento:
  bytes → librosa.load → noisereduce → trim → parselmouth features
"""

import io
import warnings
import numpy as np
import librosa
import noisereduce as nr

warnings.filterwarnings("ignore")

SR     = 16_000
TOP_DB = 20


def preprocess_waveform(audio_bytes: bytes) -> np.ndarray:
    """Bytes de audio crudo → waveform normalizado (mono, 16 kHz, sin ruido, trimmed)."""
    y, _ = librosa.load(io.BytesIO(audio_bytes), sr=SR, mono=True)
    y    = nr.reduce_noise(y=y, sr=SR)
    y, _ = librosa.effects.trim(y, top_db=TOP_DB)
    return y


def extract_acoustic_features(audio_bytes: bytes, feature_names: list[str]) -> dict | None:
    """
    Extrae las features acústicas pedidas con parselmouth (misma clase que en entrenamiento).
    Devuelve {feature_name: value} o None si el audio es inválido.
    """
    from src.features.acoustic import ExtraccionCaracteristicas

    try:
        y = preprocess_waveform(audio_bytes)

        if len(y) < int(SR * 0.5):
            return None

        extractor = ExtraccionCaracteristicas()
        all_feats = extractor.extraer_desde_array(y, SR)
        if all_feats is None:
            return None

        return {f: all_feats.get(f) for f in feature_names if f in all_feats}

    except Exception as e:
        print(f"[extractor] {e}")
        return None
