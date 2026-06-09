# Multi-stage Docker build for Lung Cancer Diagnostic System

# Base stage with Python and system dependencies
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && mkdir -p /app \
    && chown -R app:app /app

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM base as production

# Copy application code
COPY --chown=app:app . .

# Create necessary directories
RUN mkdir -p /app/static/uploads /app/models /app/logs \
    && chown -R app:app /app/static /app/models /app/logs

# Create volume mount points
VOLUME ["/app/models", "/app/logs", "/app/static/uploads"]

# Switch to non-root user
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Expose port
EXPOSE 5000

# Start application
CMD ["python", "src/web/app.py"]

# Development stage
FROM base as development

# Install additional development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    flake8 \
    black \
    isort \
    mypy

# Copy application code
COPY --chown=app:app . .

# Create necessary directories
RUN mkdir -p /app/static/uploads /app/models /app/logs \
    && chown -R app:app /app/static /app/models /app/logs

# Switch to non-root user
USER app

# Expose port
EXPOSE 5000

# Default command for development
CMD ["python", "src/web/app.py"]

# Testing stage
FROM development as testing

# Copy test files
COPY --chown=app:app tests/ ./tests/

# Run tests
CMD ["pytest", "tests/", "-v", "--cov=src", "--cov-report=xml"]</content>
<parameter name="filePath">c:\Users\Anand Singh\OneDrive\Desktop\Major\Dockerfile