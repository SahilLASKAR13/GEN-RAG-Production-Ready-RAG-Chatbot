# deployment/Dockerfile

# -----------------------------
# Base Image
# -----------------------------
FROM python:3.12-slim

# -----------------------------
# Set Working Directory
# -----------------------------
WORKDIR /app

# -----------------------------
# Environment Variables
# -----------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -----------------------------
# Install System Dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Copy Requirements
# -----------------------------
COPY requirements.txt .

# -----------------------------
# Upgrade pip
# -----------------------------
RUN pip install --upgrade pip

# -----------------------------
# Install Python Dependencies
# -----------------------------
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy Entire Project
# -----------------------------
COPY . .

# -----------------------------
# Create Required Directories
# -----------------------------
RUN mkdir -p storage/faiss_index
RUN mkdir -p data/uploads

# -----------------------------
# Expose FastAPI Port
# -----------------------------
EXPOSE 8000

# -----------------------------
# Run FastAPI
# -----------------------------
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]