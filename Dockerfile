# syntax=docker/dockerfile:1
# Self-contained image: the search index is built at image build time -> instant, offline startup.

# ---------- builder: install deps, download ONNX models, build the index ----------
FROM python:3.12-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/hf FASTEMBED_CACHE_PATH=/app/.cache/fastembed HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ app/
COPY data/processed/ data/processed/
# embeddings come from data/processed/embeddings_cache.npz when present (fast); otherwise computed here
RUN python -m app.build_index

# ---------- runtime ----------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/app/.cache/hf FASTEMBED_CACHE_PATH=/app/.cache/fastembed HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
# Hugging Face Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=user:user /app/.cache /app/.cache
COPY --from=builder --chown=user:user /app/index /app/index
COPY --chown=user:user app/ app/
COPY --chown=user:user data/processed/scrape_report.json data/processed/scrape_report.json
RUN mkdir -p /app/logs && chown user:user /app/logs

USER user
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/api/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*", "--timeout-keep-alive", "30"]
