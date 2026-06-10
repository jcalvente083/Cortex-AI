# Diagrama de Secuencia — Cortex-AI

PlantUML para el diagrama de secuencia del caso de uso principal (`POST /predict/batch`).

## Cómo renderizarlo

- **Online:** pegar el bloque en [plantuml.com/plantuml](https://www.plantuml.com/plantuml/uml/)
- **VS Code:** extensión `PlantUML` (requiere Java)
- **CLI:** `java -jar plantuml.jar diagrama_secuencia.md` → genera PNG

Una vez exportado como PNG, guardarlo en `docs/Memoria/imagenes/anexos/diagrama_secuencia.png` y reemplazar en `software.tex` el placeholder por:
```latex
\includegraphics[width=0.9\textwidth]{imagenes/anexos/diagrama_secuencia.png}
```

---

## Código PlantUML

```plantuml
@startuml Cortex-AI — Diagrama de Secuencia: POST /predict/batch

skinparam sequenceMessageAlign center
skinparam responseMessageBelowArrow true
skinparam maxMessageSize 160
skinparam BoxPadding 10

actor  "Facultativo\nMédico"          as user
participant "App Flutter\n(Móvil)"    as app   #LightBlue
participant "FastAPI\nRaspberry Pi 5" as api   #LightGreen
participant "Modal.com\nGPU Cloud"    as cloud #LightYellow

== Configuración del servidor (CU-01 / CU-06) ==

user  ->  app  : Introduce URL del servidor\ny pulsa "Guardar"
app   ->  api  : GET /health
api  -->  app  : HealthResponse\n{status: ok, modelos: [knn, xgboost, resnet18, …]}
app  -->  user : ✓ Servidor conectado (SnackBar verde)

== Sesión de grabación (CU-03) ==

user  ->  app  : Pulsa "Iniciar Análisis"
app  -->  user : Paso 1 — Vocal sostenida (/a/ prolongada)
user  ->  app  : Graba y confirma audio vocal
app  -->  user : Paso 2 — Lee frase (banco de 12, aleatoria)
user  ->  app  : Graba y confirma audio de frase
app  -->  user : Paso 3 — Describe imagen de escena cotidiana
user  ->  app  : Graba y confirma audio espontáneo

== Envío y predicción (CU-04) ==

app   ->  api  : POST /predict/batch\nmultipart {audio_vocal, audio_frase,\naudio_espontanea, modelo}

alt modelo local (knn | xgboost | resnet18)

    loop actividad ∈ {vocal, frase, espontanea}
        api   ->  api  : Preprocesar audio\n(librosa: 16 kHz mono, trim silencio)
        api   ->  api  : Extraer features acústicas\n(Praat: jitter, shimmer, HNR, CPPS, ATRI)
        note right api
            ResNet18: genera mel-espectrograma
            (ventana 1s, hop 0.5s, 65 mels)
            y propaga por frames → agrega probs
        end note
        api   ->  api  : Inferencia modelo local\n→ probabilidad_pd[actividad]
        api   ->  api  : Calcular explicabilidad\n(SHAP / Feature Deviation / Grad-CAM)
    end

else modelo cloud (wav2vec_embeddings | wav2vec_finetune)

    loop actividad ∈ {vocal, frase, espontanea}
        api   ->  api  : Preprocesar audio\n(16 kHz mono, truncar/pad a 10 s)
        api   ->  cloud : POST /predict (httpx async)\n{audio_bytes, actividad}
        cloud ->  cloud : Wav2Vec2 XLSR-53 español\n→ embeddings 1024-dim
        cloud ->  cloud : PCA (95 % varianza)\n+ clasificador (XGBoost / KNN)
        cloud -->  api  : {probabilidad_pd}
    end

end

api   ->  api  : Promediar probabilidades (vocal + frase + espontanea) / 3\ncalcular nivel_riesgo con umbral óptimo τ
api  -->  app  : BatchPrediccionResponse\n{prob_final, prediccion, nivel_riesgo,\ndetalle_por_actividad, explicabilidad,\ngrad_cam_por_actividad}

== Presentación del resultado (CU-05) ==

app  -->  user : ResultScreen — RiskGauge (verde/amarillo/rojo)
app  -->  user : "¿Por qué este resultado?"\nbarras SHAP / Feature Deviation\no imágenes Grad-CAM por actividad

@enduml
```
