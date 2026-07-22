# ============================================================
# Dockerfile — Cattle Weight Estimation API
# Multi-stage build with CPU-only PyTorch
# ============================================================

FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

RUN sed -i 's/http:/https:/g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/base.txt ./requirements.txt
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ------------------------------------------------------------

FROM python:3.11-slim-bookworm AS runtime

RUN sed -i 's/http:/https:/g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /install/lib/python3.11/site-packages \
     /usr/local/lib/python3.11/site-packages
COPY --from=builder /install/bin /usr/local/bin

COPY src/ ./src/
COPY models/ ./models/

ENV PYTHONUNBUFFERED=1

RUN useradd --no-create-home --shell /bin/false appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 4001

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "4001", "--workers", "1", "--loop", "uvloop"]
