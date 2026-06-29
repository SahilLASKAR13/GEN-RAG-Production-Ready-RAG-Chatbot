# deployment/Dockerfile

FROM python:3.11-slim

WORKDIR /app

# System deps for pypdf and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]
