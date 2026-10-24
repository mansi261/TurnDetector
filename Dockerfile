FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

COPY requirements-serving.txt ./
RUN python -m pip install --no-cache-dir --retries 10 --timeout 120 \
    -r requirements-serving.txt

ENV TURN_DETECTOR_CHECKPOINT=/app/model/model.fp32.onnx \
    TURN_DETECTOR_METADATA=/app/model/model_metadata.json \
    TURN_DETECTOR_MODEL_MANIFEST=/app/model/serving_models.json \
    TURN_DETECTOR_MODEL_VERSION=english-whisper-tiny-baseline-v0 \
    TURN_DETECTOR_THRESHOLD=0.5 \
    TURN_DETECTOR_ORT_INTRA_OP_THREADS=2 \
    TURN_DETECTOR_ORT_INTER_OP_THREADS=1 \
    TURN_DETECTOR_ORT_EXECUTION_MODE=sequential

COPY src ./src
COPY model ./model

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3)"
CMD ["uvicorn", "turn_detector.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
