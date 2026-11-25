FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MODEL_CACHE_DIR=/models/speechbrain-ecapa \
    DATABASE_URL=sqlite:////data/speakers.db

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /service
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt
COPY app ./app
RUN mkdir -p /data /models/speechbrain-ecapa && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /service /data /models
USER appuser
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
