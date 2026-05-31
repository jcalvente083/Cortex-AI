import pandas as pd

# Cambia la ruta a tu CSV si es distinta
df = pd.read_csv('data/processed/combined/dataset_combinado_raw.csv')

# Para asegurarnos de que contamos pacientes únicos (y no grabaciones repetidas)
pacientes = df.drop_duplicates(subset=['ID_Paciente'])

print("--- DEMOGRAFÍA POR DATASET Y CLASE ---")
for dataset in ['NeuroVoz', 'PC-GITA']:
    for target in [0, 1]:
        subset = pacientes[(pacientes['Dataset'] == dataset) & (pacientes['Target'] == target)]
        hombres = len(subset[subset['Sex'] == 0]) # O 'H', según lo tengas
        mujeres = len(subset[subset['Sex'] == 1]) # O 'M', según lo tengas
        edad_media = subset['Age'].mean()
        edad_std = subset['Age'].std()
        clase = "Control (0)" if target == 0 else "Parkinson (1)"
        print(f"{dataset} - {clase}: Total={len(subset)}, H={hombres}, M={mujeres}, Edad={edad_media:.2f} ± {edad_std:.2f}")

print("\n--- MEDIAS DE LAS CARACTERÍSTICAS ACÚSTICAS ---")
features = ['rPPQ', 'ShimmerDb', 'Hnr', 'ATRI', 'CHNR']
# Las agrupamos por Target (0=Control, 1=Parkinson) usando TODAS las grabaciones
resumen_features = df.groupby('Target')[features].agg(['mean', 'std'])
print(resumen_features)