FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema (librosa necesita libsndfile, ffmpeg para decode)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    build-essential \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU — índice separado (ARM64 no tiene CUDA)
RUN pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu \
    --no-cache-dir

# Resto de dependencias del backend
COPY deploy/requirements-rpi.txt .
RUN pip install -r requirements-rpi.txt --no-cache-dir

# Código fuente (backend + src que necesita extractor.py)
COPY backend/ backend/
COPY src/ src/

# Los modelos se montan como volumen en runtime (ver docker-compose.yml)

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
