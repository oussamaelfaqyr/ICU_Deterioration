# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: dependency builder
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile some packages (lightgbm, shap)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements-api.txt

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# libgomp is required at runtime by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application code
COPY api.py promote_model.py drift_detector.py ./

# Copy pre-built artefacts (preprocessing pipeline + test data for SHAP background)
# These are baked into the image; override with a volume mount in docker-compose for dev
COPY mimic_processed/ ./mimic_processed/
COPY mlflow.db ./mlflow.db
COPY mlruns/ ./mlruns/
COPY patch_mlflow_paths.py ./patch_mlflow_paths.py

# Patch the MLflow SQLite DB with Docker absolute paths
RUN python patch_mlflow_paths.py

# Create dirs for runtime files
RUN mkdir -p /app/logs

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health-check so Docker / compose can detect when the API is ready
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
