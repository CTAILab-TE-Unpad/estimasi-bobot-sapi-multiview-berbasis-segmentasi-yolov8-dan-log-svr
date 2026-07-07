# ============================================================
# Dockerfile — Cattle Weight Estimation API
# Multi-stage build with CPU-only PyTorch (~1.0–1.3 GB final)
# ============================================================

# ------------------------------------------------------------
# Stage 1: builder — install all Python dependencies
# ------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — separate layer for better cache reuse
COPY requirements/base.txt ./requirements.txt

# Install to a prefix directory so we can cleanly copy to runtime stage
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ------------------------------------------------------------
# Stage 2: runtime — minimal final image
# ------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Runtime system libraries required by OpenCV and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install/lib/python3.11/site-packages \
     /usr/local/lib/python3.11/site-packages
COPY --from=builder /install/bin /usr/local/bin

# Copy application source and model artifacts
COPY src/ ./src/
COPY models/ ./models/
COPY main.py .

# Make src/ importable (alternative to installing the package)
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Non-root user for security
RUN useradd --no-create-home --shell /bin/false appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Single worker — see main.py for explanation
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
