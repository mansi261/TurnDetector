FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    TURN_DETECTOR_CHECKPOINT=/app/model/model.fp32.onnx \
    TURN_DETECTOR_METADATA=/app/model/model_metadata.json \
    TURN_DETECTOR_MODEL_VERSION=english-whisper-tiny-baseline-v0 \
    TURN_DETECTOR_THRESHOLD=0.5

COPY requirements-serving.txt ./
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY src ./src
COPY model ./model

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3)"
CMD ["uvicorn", "turn_detector.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
