# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash drcrop

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/uploads /app/models /app/logs /app/database /app/data && \
    chown -R drcrop:drcrop /app

# Copy application code
COPY --chown=drcrop:drcrop . .

# Switch to non-root user
USER drcrop

# Create a startup script
COPY --chown=drcrop:drcrop <<EOF /app/start.sh
#!/bin/bash
set -e

echo "Starting DrCrop AI Crop Disease Detector..."

# Check if model exists, if not create sample
if [ ! -f "/app/models/trained_model.h5" ]; then
    echo "No trained model found. Creating sample model structure..."
    python -c "
from models.crop_disease_model import CropDiseaseDetector
detector = CropDiseaseDetector()
detector.build_model()
print('Sample model architecture created')
"
fi

# Initialize database if not exists
if [ ! -f "/app/database/disease_info.json" ]; then
    echo "Initializing disease database..."
    python -c "
from database.disease_database import create_database
create_database()
print('Disease database initialized')
"
fi

# Start the application
echo "Starting FastAPI application..."
exec uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
EOF

RUN chmod +x /app/start.sh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Start the application
CMD ["/app/start.sh"]

# Labels for metadata
LABEL maintainer="DrCrop Team"
LABEL version="1.0.0"
LABEL description="AI-powered crop disease detection system"
LABEL org.opencontainers.image.source="https://github.com/drcrop/ai-crop-detector"