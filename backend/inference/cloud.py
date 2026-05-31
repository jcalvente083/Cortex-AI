"""
cloud.py — Proxy asíncrono hacia la API Modal (Wav2Vec2).
"""

import httpx
from backend.config import CLOUD_API_URL


async def predict_cloud(audio_bytes: bytes, modelo: str, actividad: str) -> dict:
    """
    Reenvía el audio a la API cloud y devuelve el dict de respuesta.
    Lanza RuntimeError si CLOUD_API_URL no está configurada.
    Lanza httpx.HTTPStatusError si el cloud devuelve error HTTP.
    """
    if not CLOUD_API_URL:
        raise RuntimeError(
            "Modelos Wav2Vec2 no disponibles en este dispositivo. "
            "Configura la variable de entorno CLOUD_API_URL."
        )

    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            f"{CLOUD_API_URL}/predict",
            files={"audio": ("audio.wav", audio_bytes, "audio/wav")},
            data={"modelo": modelo, "actividad": actividad},
        )
        response.raise_for_status()
        return response.json()
