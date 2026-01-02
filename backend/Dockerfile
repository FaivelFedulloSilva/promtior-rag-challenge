FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY requirements.api.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
docker tag promtior-api:latest 258003046321.dkr.ecr.us-east-1.amazonaws.com/promtior-api:latest
docker push 258003046321.dkr.ecr.us-east-1.amazonaws.com/promtior-api:latest
aws configure get region
docker tag promtior-api:latest 258003046321.dkr.ecr.us-east-1.amazonaws.com/promtior-api:latest
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/promtior-api:latest


docker tag promtior-fullstack:latest 258003046321.dkr.ecr.us-east-1.amazonaws.com/promtior-api:latest
docker push 258003046321.dkr.ecr.us-east-1.amazonaws.com/promtior-api:latest