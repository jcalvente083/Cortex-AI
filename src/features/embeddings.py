"""
Extrae embeddings de Wav2Vec2 para todos los audios del dataset combinado.

Entrada : data/processed/combined/dataset_combinado_raw.csv
Salida  : data/processed/combined/embeddings_wav2vec.csv
Caché   : data/processed/combined/cache_embeddings.csv

Modelo  : jonatasgrosman/wav2vec2-large-xlsr-53-spanish
          Mean pooling sobre last_hidden_state → vector 1024-dim por audio.
"""

import os
import sys
import warnings

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import librosa
import numpy as np
import pandas as pd
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from tqdm import tqdm

from src.config import SR

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_CSV = "data/processed/combined/dataset_combinado_raw.csv"
OUT_DIR   = "data/processed/combined"
CACHE_CSV = f"{OUT_DIR}/cache_embeddings.csv"
FINAL_CSV = f"{OUT_DIR}/embeddings_wav2vec.csv"

META_COLS   = ['audio_path', 'ID_Paciente', 'Dataset', 'Target', 'Sex', 'Age', 'activity_type']
MIN_SAMPLES = int(SR * 0.1)   # 100 ms mínimo


# =============================================================================
# MODELO
# =============================================================================

def load_model():
    print(f"Cargando {MODEL_NAME} ...")
    print(f"Dispositivo : {DEVICE.upper()}")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    print(f"Modelo cargado  — dim embeddings: {model.config.hidden_size}\n")
    return model, feature_extractor


# =============================================================================
# EXTRACCIÓN
# =============================================================================

def _extract_embedding(audio_path: str, model, feature_extractor) -> np.ndarray | None:
    try:
        waveform, _ = librosa.load(audio_path, sr=SR, mono=True)

        if len(waveform) < MIN_SAMPLES:
            return None

        inputs = feature_extractor(
            waveform,
            sampling_rate=SR,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(DEVICE)

        with torch.no_grad():
            outputs = model(input_values)

        # Mean pooling sobre la dimensión temporal → (1024,)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding

    except Exception as e:
        print(f"  [x] {os.path.basename(audio_path)}: {e}")
        return None


# =============================================================================
# CACHÉ
# =============================================================================

def _load_cache():
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(CACHE_CSV):
        df   = pd.read_csv(CACHE_CSV)
        done = set(df['audio_path'].tolist())
        print(f"Caché: {len(done)} embeddings previos.")
        return df, done
    return pd.DataFrame(), set()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 62)
    print("  EXTRACCIÓN DE EMBEDDINGS — Wav2Vec2 ES")
    print("=" * 62)

    if not os.path.exists(INPUT_CSV):
        print(f"[!] No encontrado: {INPUT_CSV}")
        print("    Ejecuta primero: uv run python -m src.data.build_combined_dataset")
        return

    df_input = pd.read_csv(INPUT_CSV)
    print(f"Dataset : {len(df_input)} grabaciones | {df_input['ID_Paciente'].nunique()} pacientes")

    cache_df, already_done = _load_cache()

    pending = df_input[~df_input['audio_path'].isin(already_done)]
    print(f"Pendientes : {len(pending)} | Ya procesados : {len(already_done)}\n")

    if len(pending) > 0:
        model, feature_extractor = load_model()

        new_rows = []
        errors   = []

        for _, row in tqdm(pending.iterrows(), total=len(pending), desc="Embeddings"):
            emb = _extract_embedding(row['audio_path'], model, feature_extractor)
            if emb is None:
                errors.append(row['audio_path'])
                continue

            record = {col: row[col] for col in META_COLS}
            for i, val in enumerate(emb):
                record[f"emb_{i}"] = val
            new_rows.append(record)

        if new_rows:
            df_new   = pd.DataFrame(new_rows)
            cache_df = pd.concat([cache_df, df_new], ignore_index=True) if len(cache_df) else df_new
            cache_df.to_csv(CACHE_CSV, index=False)
            print(f"\nCaché actualizado : {len(cache_df)} registros → {CACHE_CSV}")

        if errors:
            print(f"  Errores          : {len(errors)} archivos")

    # CSV final — solo audios presentes en el dataset actual
    valid_paths = set(df_input['audio_path'].tolist())
    df_final    = cache_df[cache_df['audio_path'].isin(valid_paths)].copy()
    df_final.to_csv(FINAL_CSV, index=False)

    emb_dim = sum(1 for c in df_final.columns if c.startswith("emb_"))
    print(f"\n{'='*62}")
    print(f"  EMBEDDINGS FINALES : {len(df_final)} grabaciones")
    print(f"  Pacientes únicos   : {df_final['ID_Paciente'].nunique()}")
    print(f"  Dimensión          : {emb_dim}")
    print(f"{'='*62}")
    print(df_final.groupby(['Dataset', 'Target', 'activity_type']).size()
          .rename('n').reset_index().to_string(index=False))
    print(f"\n  Guardado en : {FINAL_CSV}")


if __name__ == "__main__":
    main()
