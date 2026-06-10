"""
schemas.py — Modelos Pydantic de request y response.
"""

from typing import Optional
from pydantic import BaseModel


class FeatureInfo(BaseModel):
    nombre:      str
    valor:       float
    contribucion: float   # SHAP value o z-score normalizado
    direccion:   str      # "positivo" (→PD) | "negativo" (→HC) | "neutro"


class Explicabilidad(BaseModel):
    disponible:  bool
    tipo:        Optional[str] = None         # "shap" | "feature_deviation" | None
    features:    list[FeatureInfo] = []
    base_value:  Optional[float]  = None      # solo SHAP


class PrediccionResponse(BaseModel):
    probabilidad_pd: float    # [0.0 – 1.0]
    prediccion:      str      # "Sano" | "Parkinson"
    nivel_riesgo:    str      # "Bajo" | "Medio" | "Alto"
    umbral:          float
    modelo:          str
    actividad:       str
    explicabilidad:  Explicabilidad


class BatchPrediccionResponse(BaseModel):
    probabilidad_pd_final:    float
    prediccion:               str
    nivel_riesgo:             str
    umbral_promedio:          float
    modelo:                   str
    detalle_por_actividad:    dict[str, float]          # actividad → prob
    explicabilidad:           Optional[Explicabilidad] = None
    grad_cam_por_actividad:   Optional[dict[str, Optional[str]]] = None  # actividad → base64 PNG


class ModeloInfo(BaseModel):
    nombre:          str
    disponible:      bool
    actividades:     list[str]
    explicabilidad:  bool
    ubicacion:       str       # "local" | "cloud"
    descripcion:     str


class HealthResponse(BaseModel):
    status:            str
    cloud_disponible:  bool
    modelos:           list[ModeloInfo]
