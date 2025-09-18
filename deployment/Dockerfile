# Multi-stage Dockerfile for Healthcare Recommendation System

# Backend Stage
FROM python:3.9-slim as backend

WORKDIR /app/backend

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ .
COPY data/ ../data/

# Expose backend port
EXPOSE 8000

# Command to run backend
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# Frontend Stage
FROM python:3.9-slim as frontend

WORKDIR /app/frontend

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Streamlit and dependencies
RUN pip install streamlit==1.28.1 plotly==5.17.0 pandas==2.1.3 requests==2.31.0

# Copy frontend code
COPY frontend/ .

# Expose frontend port
EXPOSE 8501

# Command to run frontend
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]

# Production Stage (combines both)
FROM python:3.9-slim as production

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install all dependencies
COPY backend/requirements.txt backend_requirements.txt
COPY deployment/requirements.txt deployment_requirements.txt
RUN pip install --no-cache-dir -r backend_requirements.txt -r deployment_requirements.txt

# Copy application code
COPY backend/ backend/
COPY frontend/ frontend/
COPY data/ data/

# Copy supervisor configuration
COPY deployment/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose both ports
EXPOSE 8000 8501

# Use supervisor to run both services
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
