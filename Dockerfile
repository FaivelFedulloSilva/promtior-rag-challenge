# ---------- 1) Build frontend ----------
FROM node:20-slim AS frontend_builder
WORKDIR /frontend

# Copiamos manifests primero (cache)
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci

# Copiamos el resto y buildeamos
COPY frontend/ ./
RUN npm run build


# ---------- 2) Backend runtime ----------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# deps del sistema (mínimo razonable)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/*

# requirements del backend
COPY backend/requirements.api.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# copiamos backend code
COPY backend/app /app/app

# copiamos data si la necesitas en runtime (ej: chroma_db, pdfs, etc)
COPY backend/chroma_db /app/chroma_db
COPY backend/data /app/data

# ✅ copiamos el frontend compilado AL PATH que server.py espera:
# server.py -> Path(__file__).resolve().parents[2] / "frontend" / "dist"
# con /app/app/server.py => parents[2] = /app => /app/frontend/dist
COPY --from=frontend_builder /frontend/dist /app/frontend/dist

EXPOSE 8000

# IMPORTANT: host 0.0.0.0
CMD ["python", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
