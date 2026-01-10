# Dockerfile for MFT-project ML Inference
# Builds a reproducible Docker image for offline trading predictions

# Base image with Python
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source code
COPY tensortrade/ ./tensortrade/
COPY pipelines/ ./pipelines/
COPY conf/ ./conf/
COPY src/ ./src/

# Copy trained model (must run dvc pull before building)
COPY outputs/model/checkpoints/latest/ ./outputs/model/checkpoints/latest/

# Create data directory for volume mounting
RUN mkdir -p /data

# Set Python path
ENV PYTHONPATH=/app

# Default entrypoint for inference
ENTRYPOINT ["python", "-m", "src.predict"]

# Default command (can be overridden)
CMD ["--help"]

