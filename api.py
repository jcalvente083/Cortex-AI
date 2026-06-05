import os
import sys

directorio_actual = os.path.dirname(os.path.abspath(__file__))
if directorio_actual not in sys.path:
    sys.path.append(directorio_actual)
    
import io
import joblib
import librosa
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.A_PreprocesamientoAudios import PreprocesadorAudio
from src.B_ExtraccionCaracteristicas import ExtraccionCaracteristicas
from src.C_IntegradorDatos import IntegradorDatos
from src.config import SR, caracteristicas_globales

app = FastAPI(title="Cortex-AI. Predicción de Parkinson", version="1.0")

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELO_XGB = None
SCALER = None
EXPLAINER = None
COLUMNAS_TOP = caracteristicas_globales

@app.on_event("startup")
def cargar_modelo_inteligencia_artificial():
    """Se ejecuta al encender el servidor. Carga el modelo, el escalador y SHAP."""
    global MODELO_XGB, SCALER, EXPLAINER
    
    ruta_mod = "./models/xgboost_mixto.pkl"
    ruta_sca = "./models/xgboost_scaler_mixto.pkl"
    
    if not os.path.exists(ruta_mod) or not os.path.exists(ruta_sca):
        print("ALERTA: No se encontraron los modelos en ./models/")
        return
        
    MODELO_XGB = joblib.load(ruta_mod)
    SCALER = joblib.load(ruta_sca)

    EXPLAINER = shap.TreeExplainer(MODELO_XGB)
    print("¡API Lista! Modelo Mixto, Escalador y SHAP cargados en memoria.")


# ==========================================
# 2. EL ENDPOINT PRINCIPAL
# ==========================================
@app.post("/diagnostico")
async def realizar_diagnostico(
    audio: UploadFile = File(...), 
    edad: int = Form(...), 
    sexo: int = Form(...)
):
    """
    Recibe un audio desde Flutter junto con la edad y el sexo.
    Devuelve la probabilidad, el diagnóstico y la explicación SHAP.
    """
    if MODELO_XGB is None:
        raise HTTPException(status_code=500, detail="El modelo no está cargado en el servidor.")

    try:
        # 1. Leer los bytes del audio directamente desde la petición de Flutter y convertirlo en un stream
        audio_bytes = await audio.read()
        audio_stream = io.BytesIO(audio_bytes)
        
        audio_array, sr_original = librosa.load(audio_stream, sr=None)
        
        # 2. Preprocesar (Limpiar ruido y resamplear)
        preprocesador = PreprocesadorAudio()
        audio_limpio = preprocesador.preprocesarAudio(audio_array, sr_origen=sr_original)
        
        # 3. Extraer Biomarcadores (Praat)
        extractor = ExtraccionCaracteristicas()
        caracs = extractor.extraer_desde_array(audio_limpio, sr_origen=SR)
        
        if caracs is None:
            raise HTTPException(status_code=400, detail="Error en Praat: El audio no es procesable.")
        
        # Crear el DataFrame exacto
        unionDatos = IntegradorDatos()
        datos_paciente = {'Age': edad, 'Sex': sexo}
        df_entrada = pd.DataFrame(unionDatos.integrar_en_memoria(caracs, datos_paciente))

        # Seleccionar solo las columnas que el modelo espera
        df_entrada = df_entrada[COLUMNAS_TOP]

        # Escalar los datos
        entrada_escalada = SCALER.transform(df_entrada)
        df_escalado = pd.DataFrame(entrada_escalada, columns=COLUMNAS_TOP)

        # Predecir
        prediccion = int(MODELO_XGB.predict(entrada_escalada)[0])
        prob_parkinson = float(MODELO_XGB.predict_proba(entrada_escalada)[0][1] * 100)
        diagnostico_texto = "Parkinson" if prediccion == 1 else "Sano"

        # EXPLICABILIDAD SHAP 
        shap_values_paciente = EXPLAINER(df_escalado)
        
        explicacion_shap = {
            col: float(val) 
            for col, val in zip(COLUMNAS_TOP, shap_values_paciente.values[0])
        }

        valor_base = float(shap_values_paciente.base_values[0])

        # ==========================================
        # DEVOLVER EL JSON A FLUTTER
        # ==========================================
        return {
            "status": "success",
            "diagnostico": diagnostico_texto,
            "probabilidad_parkinson_pct": round(prob_parkinson, 2),
            "caracteristicas_extraidas": df_entrada.iloc[0].to_dict(),
            "shap_explicabilidad": {
                "valor_base_modelo": round(valor_base, 4),
                "impacto_variables": explicacion_shap
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")