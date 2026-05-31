import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.config import SEED

INPUT_CSV   = "data/processed/combined/dataset_combinado_raw.csv"
TRAIN_CSV   = "data/processed/combined/train_80.csv"
HOLDOUT_CSV = "data/processed/combined/holdout_20.csv"


def generar_particion_pacientes(df: pd.DataFrame, test_size: float = 0.20) -> tuple:
    """
    Split 80/20 a nivel de paciente, estratificado por Target × Dataset.
    Garantiza que ningún paciente aparezca en ambos splits.
    """
    pacientes = (
        df[['ID_Paciente', 'Target', 'Dataset']]
        .drop_duplicates('ID_Paciente')
        .copy()
    )

    # Clave de estratificación combinada: Target + Dataset
    pacientes['_strat'] = pacientes['Target'].astype(str) + '_' + pacientes['Dataset']

    train_ids, holdout_ids = train_test_split(
        pacientes['ID_Paciente'],
        test_size=test_size,
        stratify=pacientes['_strat'],
        random_state=SEED,
    )

    df_train   = df[df['ID_Paciente'].isin(train_ids)].copy()
    df_holdout = df[df['ID_Paciente'].isin(holdout_ids)].copy()

    # Verificación de seguridad
    intersect = set(df_train['ID_Paciente']) & set(df_holdout['ID_Paciente'])
    if intersect:
        raise ValueError(f"DATA LEAKAGE — pacientes en ambos splits: {intersect}")

    print(f"Pacientes  — train: {len(train_ids):>4}  |  holdout: {len(holdout_ids):>4}")
    print(f"Grabaciones — train: {len(df_train):>4}  |  holdout: {len(df_holdout):>4}")
    print()
    print(df_train.groupby(['Dataset', 'Target']).size().rename('train').reset_index().to_string(index=False))
    print()
    print(df_holdout.groupby(['Dataset', 'Target']).size().rename('holdout').reset_index().to_string(index=False))

    return df_train, df_holdout


def main():
    df = pd.read_csv(INPUT_CSV)

    df_train, df_holdout = generar_particion_pacientes(df)

    Path(TRAIN_CSV).parent.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(TRAIN_CSV, index=False)
    df_holdout.to_csv(HOLDOUT_CSV, index=False)

    print(f"\nGuardado: {TRAIN_CSV}")
    print(f"Guardado: {HOLDOUT_CSV}")


if __name__ == "__main__":
    main()
