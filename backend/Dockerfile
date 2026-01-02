FROM python:3.11-slim

WORKDIR /app

# deps del sistema (si llegás a necesitar)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.backend.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# ajustá el comando a tu entrypoint real
CMD ["python", "-m", "app.server"]
