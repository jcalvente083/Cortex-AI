"""
Valida audios y construye metadatos para el pipeline ResNet18.

Entrada : data/processed/combined/dataset_combinado_raw.csv
Salida  : data/processed/combined/spectrograms_meta.csv
Cache   : data/processed/combined/cache_spectrograms.csv

Para cada audio valido registra duracion y numero de frames resultantes.
El entrenamiento (train_resnet.py) genera los espectrogramas on-the-fly.
"""

import os
import sys
import warnings

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import SR

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACION — deben coincidir con train_resnet.py
# =============================================================================
FRAME_LEN_S = 0.4     # duracion de cada frame (segundos)
HOP_LEN_S   = 0.2     # paso entre frames (segundos)
N_MELS      = 65
N_FFT       = 512
N_FFT_HOP_S = 0.03    # hop del STFT interno (segundos)
MIN_DUR_S   = 0.1     # duracion minima para audio valido
SILENCE_THR = 1e-4    # umbral de amplitud para silencio

INPUT_CSV = "data/processed/combined/dataset_combinado_raw.csv"
OUT_DIR   = "data/processed/combined"
CACHE_CSV = f"{OUT_DIR}/cache_spectrograms.csv"
FINAL_CSV = f"{OUT_DIR}/spectrograms_meta.csv"

META_COLS = ['audio_path', 'ID_Paciente', 'Dataset', 'Target', 'Sex', 'Age', 'activity_type']


# =============================================================================
# VALIDACION
# =============================================================================
def _validate_audio(audio_path: str) -> dict | None:
    try:
        y, _ = librosa.load(audio_path, sr=SR, mono=True)
        duration_s = len(y) / SR

        if duration_s < MIN_DUR_S:
            return None
        if np.max(np.abs(y)) < SILENCE_THR:
            return None

        frame_len = int(SR * FRAME_LEN_S)
        hop_len   = int(SR * HOP_LEN_S)
        n_frames  = max(1, (len(y) - frame_len) // hop_len + 1) if len(y) >= frame_len else 1

        return {'duration_s': round(duration_s, 3), 'n_frames': n_frames}

    except Exception as e:
        print(f"  [x] {os.path.basename(audio_path)}: {e}")
        return None


# =============================================================================
# CACHE
# =============================================================================
def _load_cache() -> tuple[pd.DataFrame, set]:
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(CACHE_CSV):
        df   = pd.read_csv(CACHE_CSV)
        done = set(df['audio_path'].tolist())
        print(f"Cache: {len(done)} audios previos.")
        return df, done
    return pd.DataFrame(), set()


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 62)
    print("  VALIDACION DE AUDIOS — Pipeline ResNet18")
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

    new_rows, errors = [], []
    for _, row in tqdm(pending.iterrows(), total=len(pending), desc="Validando audios"):
        info = _validate_audio(row['audio_path'])
        if info is None:
            errors.append(row['audio_path'])
            continue
        record = {col: row[col] for col in META_COLS if col in row.index}
        record.update(info)
        record['valid'] = True
        new_rows.append(record)

    if new_rows:
        df_new   = pd.DataFrame(new_rows)
        cache_df = pd.concat([cache_df, df_new], ignore_index=True) if len(cache_df) else df_new
        cache_df.to_csv(CACHE_CSV, index=False)
        print(f"\nCache actualizado: {len(cache_df)} registros -> {CACHE_CSV}")

    if errors:
        print(f"  Audios descartados: {len(errors)}")

    valid_paths = set(df_input['audio_path'].tolist())
    df_final = cache_df[cache_df['audio_path'].isin(valid_paths) & cache_df['valid']].copy()
    df_final.to_csv(FINAL_CSV, index=False)

    total_frames = int(df_final['n_frames'].sum())
    print(f"\n{'='*62}")
    print(f"  METADATOS FINALES : {len(df_final)} audios validos")
    print(f"  Pacientes unicos  : {df_final['ID_Paciente'].nunique()}")
    print(f"  Total frames      : {total_frames:,}")
    print(f"  Duracion media    : {df_final['duration_s'].mean():.1f}s por audio")
    print(f"{'='*62}")
    print(df_final.groupby(['Dataset', 'Target', 'activity_type'])
          .agg(audios=('audio_path', 'count'), frames=('n_frames', 'sum'))
          .reset_index().to_string(index=False))
    print(f"\n  Guardado en : {FINAL_CSV}")


if __name__ == "__main__":
    main()
