tfg-parkinson-voice/
│
├── data/                   # ¡Añadir al .gitignore! No subas los audios al repo.
│   ├── raw/                # Audios originales de NeuroVoz y PC-GITA sin alterar.
│   ├── interim/            # Audios preprocesados (normalizados, recortados, sin ruido).
│   └── processed/          # Datasets finales tabulares (CSVs) y tensores/embeddings.
│
├── models/                 # ¡Añadir al .gitignore! Pesos y modelos entrenados.
│   ├── traditional/        # Modelos guardados de KNN y XGBoost (ej. .pkl, .joblib).
│   ├── resnet/             # Pesos del Fine-Tuning de ResNet18 (ej. .pth, .pt).
│   └── wav2vec/            # Pesos del Fine-Tuning y extractor de embeddings.
│
├── notebooks/              # Jupyter Notebooks para exploración y visualización.
│   ├── 01_eda_neurovoz_pcgita.ipynb
│   └── 02_feature_selection.ipynb
│
├── src/                    # Código fuente del pipeline de Machine Learning (Python).
│   ├── __init__.py
│   ├── config.py           # Variables globales (rutas, sample_rate, semillas aleatorias).
│   ├── data/               # Scripts para limpiar y normalizar audios.
│   ├── features/           # Extracción de características.
│   │   ├── acoustic.py     # Funciones para extraer features (MFCCs, jitter, shimmer).
│   │   ├── spectrograms.py # Generación de mel-espectrogramas (para ResNet18).
│   │   └── embeddings.py   # Script para obtener embeddings con Wav2Vec 2.0.
│   ├── models/             # Scripts de entrenamiento y validación.
│   │   ├── train_knn.py
│   │   ├── train_xgboost.py
│   │   ├── train_resnet.py
│   │   └── train_wav2vec.py
│   └── evaluation/         # Scripts para calcular métricas (Accuracy, F1, ROC-AUC) y gráficas.
│
├── backend/                # API REST (FastAPI).
│   ├── requirements.txt    # Dependencias exclusivas de la API.
│   ├── main.py             # Punto de entrada de FastAPI.
│   ├── api/                # Endpoints (ej. /predict).
│   ├── core/               # Configuración, seguridad, manejo de errores.
│   └── services/           # Lógica de negocio (carga de modelos y ejecución de inferencias).
│
├── frontend/               # Aplicación móvil (Flutter).
│   ├── lib/                # Código Dart (UI, lógicas de estado, llamadas HTTP a la API).
│   ├── pubspec.yaml        # Dependencias de Flutter (ej. record, http, provider/riverpod).
│   ├── android/
│   └── ios/
│
├── docs/                   # Documentación del TFG.
│   ├── memoria/            # Archivos LaTeX o Word de tu memoria.
│   └── figures/            # Diagramas de arquitectura, matrices de confusión, etc.
│
├── .gitignore
└── README.md               # Instrucciones generales de cómo levantar el entorno.