# ── Autonomous Financial Analyst — Backend + Frontend ────────────────────────
# Multi-stage: builder installs deps, runtime image is lean
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock .
RUN pip install --no-cache-dir --prefix=/install -r requirements.lock

# ── Runtime image ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# Create data directory (SQLite DB + FAISS index land here)
RUN mkdir -p data

# Expose backend port
EXPOSE 8000

# Default: run the FastAPI backend
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
