"""
Construye el dataset combinado NeuroVoz + PC-GITA.

Tareas incluidas:
  - vocal      : vocales sostenidas (A1..U3 en NV; A/E/I/O/U en PG)
  - frase      : palabras/frases leídas (13 palabras NV; sentences/read text PG)
  - espontanea : habla espontánea (ESPONTANEA en NV; monologue en PG)

Tareas excluidas:
  - DDK (pa-pa-pa, ta-ta-ta, pataka…): segmentos cortos, jitter/shimmer poco fiable
  - Palabras aisladas (PC-GITA Words): igual razón

Lee de data/interim/ (audios ya preprocesados).
Salida: data/processed/combined/dataset_combinado_raw.csv
"""

import os
import re
import sys
import warnings

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import librosa

from src.features.acoustic import ExtraccionCaracteristicas

warnings.filterwarnings('ignore')

# =============================================================================
# RUTAS
# =============================================================================
NV_AUDIOS  = "data/interim/NeuroVoz/audios"
NV_META_HC = "data/raw/NeuroVoz/metadata/data_hc.csv"
NV_META_PD = "data/raw/NeuroVoz/metadata/data_pd.csv"
PG_ROOT    = "data/interim/PC-GITA"
PG_META    = "data/raw/PC-GITA/PCGITA_metadata.xlsx"
OUT_DIR    = "data/processed/combined"
CACHE_CSV  = f"{OUT_DIR}/cache_extraccion.csv"
FINAL_CSV  = f"{OUT_DIR}/dataset_combinado_raw.csv"

# Todas las features acústicas extraídas por acoustic.py (GNE excluido — siempre None)
FEATURES_ALL = [
    'JITA', 'rJitter', 'RAP', 'rPPQ', 'rSPPQ',
    'ShimmerDb', 'Shimmer', 'rAPQ', 'rSAPQ',
    'Hnr', 'Nne', 'CHNR',
    'FTRI', 'ATRI', 'FFTR', 'FATR',
]
# Subset estable para filtrar filas inválidas (features menos susceptibles a fallar en clips cortos)
FEATURES_CORE = ['ShimmerDb', 'ATRI', 'Hnr', 'CHNR', 'rPPQ']

# =============================================================================
# MAPAS DE ACTIVIDAD — NEUROVOZ
# =============================================================================
NV_VOCAL      = {f'{v}{n}' for v in 'AEIOU' for n in range(1, 4)}
NV_FRASE      = {'ABLANDADA', 'ACAMPADA', 'BARBAS', 'BURRO', 'CALLE',
                 'CARMEN', 'DIABLO', 'GANGA', 'MANGA', 'PERRO',
                 'PIDIO', 'SOMBRA', 'TOMAS'}
NV_ESPONTANEA = {'ESPONTANEA'}

# =============================================================================
# EXTRACTOR
# =============================================================================
extractor = ExtraccionCaracteristicas()


def _load_cache():
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(CACHE_CSV):
        df = pd.read_csv(CACHE_CSV)
        done = set(df['audio_path'].tolist())
        print(f"Cache: {len(done)} registros previos.")
        return df, done
    return pd.DataFrame(), set()


def _extract(audio_path: str):
    """Carga y extrae características de un audio ya preprocesado."""
    try:
        senal, sr = librosa.load(audio_path, sr=None)
        if len(senal) < 1600:
            return None
        return extractor.extraer_desde_array(senal, sr_origen=sr)
    except Exception as e:
        print(f"  [x] {os.path.basename(audio_path)}: {e}")
        return None


# =============================================================================
# NEUROVOZ
# =============================================================================

def _nv_activity(code: str):
    if code in NV_VOCAL:      return 'vocal'
    if code in NV_FRASE:      return 'frase'
    if code in NV_ESPONTANEA: return 'espontanea'
    return None


def _nv_parse(fname: str):
    m = re.match(r'^(HC|PD)_([A-Z0-9]+)_(\d{4})\.wav$', fname, re.IGNORECASE)
    if not m:
        return None
    return m.group(1).upper(), m.group(2).upper(), int(m.group(3))


def procesar_neurovoz(cache_df, already_done):
    df_hc = pd.read_csv(NV_META_HC)
    df_pd = pd.read_csv(NV_META_PD)

    meta = {}
    for df_, group in [(df_hc, 'HC'), (df_pd, 'PD')]:
        for _, row in df_.iterrows():
            try:
                sex = int(row['Sex']) if pd.notna(row['Sex']) else -1
            except (ValueError, TypeError):
                sex = -1
            try:
                age = float(row['Age']) if pd.notna(row['Age']) else np.nan
            except (ValueError, TypeError):
                age = np.nan
            meta[(group, int(row['ID']))] = {'sex': sex, 'age': age}

    wavs = sorted(f for f in os.listdir(NV_AUDIOS) if f.lower().endswith('.wav'))
    print(f"\n[NeuroVoz] {len(wavs)} archivos .wav encontrados.")

    rows, skipped = [], 0
    for i, fname in enumerate(wavs, 1):
        if i % 500 == 0:
            print(f"  NeuroVoz: {i}/{len(wavs)}...")

        parsed = _nv_parse(fname)
        if parsed is None:
            skipped += 1
            continue

        group, activity, pid = parsed
        act_type = _nv_activity(activity)
        if act_type is None:
            skipped += 1
            continue

        info = meta.get((group, pid), {'sex': -1, 'age': np.nan})
        sex  = info['sex']
        if sex == -1:
            skipped += 1
            continue

        audio_path = os.path.join(NV_AUDIOS, fname)
        if audio_path in already_done:
            continue

        caracs = _extract(audio_path)
        if caracs is None:
            skipped += 1
            continue

        row = {
            'audio_path':    audio_path,
            'ID_Paciente':   f"NV_{group}_{pid:04d}",
            'Dataset':       'NeuroVoz',
            'Target':        1 if group == 'PD' else 0,
            'Sex':           sex,
            'Age':           info['age'],
            'activity_type': act_type,
        }
        for feat in FEATURES_ALL:
            row[feat] = caracs.get(feat, np.nan)
        rows.append(row)

    print(f"  NeuroVoz: {len(rows)} nuevos | omitidos: {skipped}")
    return rows


# =============================================================================
# PC-GITA
# =============================================================================

def _pg_patient_id(fname: str):
    m = re.match(r'^(AVPEPUDEA[C]?\d{4})', fname, re.IGNORECASE)
    return m.group(1).upper() if m else None


def _pg_collect_tasks():
    """
    Devuelve lista de (carpeta, activity_type, group_str).
    Excluye carpetas 'sobraron'/'sobran'.
    """
    configs = []

    # Vocales moduladas: modulated vowels/{hc,pd}/{A,E,I,O,U}/
    vow_root = os.path.join(PG_ROOT, 'modulated vowels')
    if os.path.isdir(vow_root):
        for g in ['hc', 'pd']:
            for v in ['A', 'E', 'I', 'O', 'U']:
                p = os.path.join(vow_root, g, v)
                if os.path.isdir(p):
                    configs.append((p, 'vocal', g.upper()))

    # Monólogo: monologue/sin normalizar/{hc,pd}/
    mono_root = os.path.join(PG_ROOT, 'monologue', 'sin normalizar')
    if os.path.isdir(mono_root):
        for g in ['hc', 'pd']:
            p = os.path.join(mono_root, g)
            if os.path.isdir(p):
                configs.append((p, 'espontanea', g.upper()))

    # Frases: sentences/{name}/sin normalizar/{HC,PD}/  (excluye 'sobraron')
    sent_root = os.path.join(PG_ROOT, 'sentences')
    if os.path.isdir(sent_root):
        for name in os.listdir(sent_root):
            sp = os.path.join(sent_root, name)
            if not os.path.isdir(sp):
                continue
            norm = os.path.join(sp, 'sin normalizar')
            if not os.path.isdir(norm):
                continue
            for gdir in os.listdir(norm):
                if any(k in gdir.lower() for k in ('sobraron', 'sobran')):
                    continue
                full = os.path.join(norm, gdir)
                if not os.path.isdir(full):
                    continue
                g = 'PD' if gdir.upper().startswith('PD') else 'HC'
                configs.append((full, 'frase', g))

    # Frases2: sentences2/{name}/non-normalized/{hc,pd}/
    sent2_root = os.path.join(PG_ROOT, 'sentences2')
    if os.path.isdir(sent2_root):
        for name in os.listdir(sent2_root):
            sp = os.path.join(sent2_root, name)
            if not os.path.isdir(sp):
                continue
            norm = os.path.join(sp, 'non-normalized')
            if not os.path.isdir(norm):
                continue
            for g in ['hc', 'pd']:
                p = os.path.join(norm, g)
                if os.path.isdir(p):
                    configs.append((p, 'frase', g.upper()))

    # Texto leído: read text/ayerfuialmedico/sin normalizar/{hc,pd}/
    read_root = os.path.join(PG_ROOT, 'read text', 'ayerfuialmedico', 'sin normalizar')
    if os.path.isdir(read_root):
        for g in ['hc', 'pd']:
            p = os.path.join(read_root, g)
            if os.path.isdir(p):
                configs.append((p, 'frase', g.upper()))

    return configs


def procesar_pcgita(cache_df, already_done):
    df_meta = pd.read_excel(PG_META)
    df_meta['ID'] = df_meta['RECODING ORIGINAL NAME'].astype(str).str.strip().str.upper()
    df_meta['sex_bin'] = df_meta['SEX'].apply(
        lambda x: 1 if str(x).strip().upper() == 'M' else 0
    )
    meta_sex = df_meta.set_index('ID')['sex_bin'].to_dict()
    meta_age = df_meta.set_index('ID')['AGE'].to_dict()

    task_configs = _pg_collect_tasks()
    print(f"\n[PC-GITA] {len(task_configs)} carpetas de tareas.")

    rows, skipped = [], 0
    total_wav = 0

    for folder, act_type, group in task_configs:
        wavs = sorted(f for f in os.listdir(folder) if f.lower().endswith('.wav'))
        total_wav += len(wavs)

        for fname in wavs:
            audio_path = os.path.join(folder, fname)
            if audio_path in already_done:
                continue

            pid = _pg_patient_id(fname)
            if pid is None or pid not in meta_sex:
                skipped += 1
                continue

            sex    = meta_sex[pid]
            age    = float(meta_age.get(pid, np.nan))
            target = 1 if group == 'PD' else 0

            caracs = _extract(audio_path)
            if caracs is None:
                skipped += 1
                continue

            row = {
                'audio_path':    audio_path,
                'ID_Paciente':   f"PG_{pid}",
                'Dataset':       'PCGITA',
                'Target':        target,
                'Sex':           sex,
                'Age':           age,
                'activity_type': act_type,
            }
            for feat in FEATURES_ALL:
                row[feat] = caracs.get(feat, np.nan)
            rows.append(row)

    print(f"  PC-GITA: {total_wav} wavs | {len(rows)} nuevos | omitidos: {skipped}")
    return rows


# =============================================================================
# PATCH AGE — añade Age a CSVs existentes sin re-extraer audio
# =============================================================================

def _patch_age(df: pd.DataFrame) -> pd.DataFrame:
    """Lee solo los metadatos y añade la columna Age al DataFrame."""
    age_map = {}

    for df_, group in [(pd.read_csv(NV_META_HC), 'HC'), (pd.read_csv(NV_META_PD), 'PD')]:
        for _, row in df_.iterrows():
            try:
                age = float(row['Age']) if pd.notna(row['Age']) else np.nan
            except (ValueError, TypeError):
                age = np.nan
            age_map[f"NV_{group}_{int(row['ID']):04d}"] = age

    df_meta = pd.read_excel(PG_META)
    df_meta['ID'] = df_meta['RECODING ORIGINAL NAME'].astype(str).str.strip().str.upper()
    for _, row in df_meta.iterrows():
        age_map[f"PG_{row['ID']}"] = float(row['AGE'])

    df = df.copy()
    df['Age'] = df['ID_Paciente'].map(age_map)
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 62)
    print("  CONSTRUCCIÓN DATASET COMBINADO — NeuroVoz + PC-GITA")
    print("=" * 62)

    cache_df, already_done = _load_cache()

    if len(cache_df) > 0 and 'Age' not in cache_df.columns:
        print("Parcheando cache con columna Age (sin re-extraccion de audio)...")
        cache_df = _patch_age(cache_df)
        cache_df.to_csv(CACHE_CSV, index=False)
        print(f"  Cache actualizada: {len(cache_df)} registros.")

    rows_nv = procesar_neurovoz(cache_df, already_done)
    rows_pg = procesar_pcgita(cache_df, already_done)

    new_rows = rows_nv + rows_pg

    if new_rows:
        df_new   = pd.DataFrame(new_rows)
        df_cache = pd.concat([cache_df, df_new], ignore_index=True) if len(cache_df) else df_new
        df_cache.to_csv(CACHE_CSV, index=False)
        print(f"\nCache actualizado: {len(df_cache)} registros -> {CACHE_CSV}")
    else:
        df_cache = cache_df
        print("\nNo hay nuevos registros; usando cache existente.")

    if len(df_cache) == 0:
        print("[!] Dataset vacío. Comprueba las rutas.")
        return

    df_final = df_cache.dropna(subset=FEATURES_CORE).copy()
    df_final = df_final[df_final['Sex'].isin([0, 1])]
    df_final.to_csv(FINAL_CSV, index=False)

    print(f"\n{'='*62}")
    print(f"  DATASET FINAL: {len(df_final)} grabaciones")
    print(f"  Pacientes únicos: {df_final['ID_Paciente'].nunique()}")
    print(f"{'='*62}")
    print(df_final.groupby(['Dataset', 'Target', 'activity_type']).size()
          .rename('n').reset_index().to_string(index=False))
    print(f"\n  Guardado en: {FINAL_CSV}")


if __name__ == "__main__":
    main()
