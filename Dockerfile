FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt


FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SENTISIS_MODEL_PATH=/app/model/artifacts/distilbert/model_int8.onnx \
    SENTISIS_TOKENIZER_PATH=/app/model/artifacts/distilbert/tokenizer/tokenizer.json

RUN useradd --create-home --uid 10001 sentisis

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY api ./api
COPY model/__init__.py model/preprocess.py model/onnx_runner.py ./model/
COPY model/artifacts/distilbert ./model/artifacts/distilbert

USER sentisis
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
