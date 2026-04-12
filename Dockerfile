FROM python:3.11-slim

# HF Spaces runs containers as uid=1000 (non-root)
RUN useradd -m -u 1000 user

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies as root first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY --chown=user:user . .

# Run as non-root for HF Spaces runtime.
USER user

# Expose HF Spaces port
EXPOSE 7860

# Health check (curl is available from the root layer)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the server bound to all interfaces on port 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
