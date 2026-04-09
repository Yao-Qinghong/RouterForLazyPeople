FROM python:3.12-slim AS base

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY router/ router/
COPY config/ config/
COPY cli.py .

# Create data directory
RUN mkdir -p /root/.llm-router/logs /root/.llm-router/metrics

# Expose default port
EXPOSE 9001

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:9001/health || exit 1

# Run the router
CMD ["uvicorn", "router.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "9001"]
