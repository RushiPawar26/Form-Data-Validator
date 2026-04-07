FROM python:3.11-slim

# System deps: OCR + PDF rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-hin \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000

# 4 worker processes × 2 threads = handles ~8 concurrent requests comfortably
# Increase WEB_CONCURRENCY env var to scale up (e.g. 8 for more users)
CMD ["sh", "-c", "gunicorn \
    --workers ${WEB_CONCURRENCY:-4} \
    --threads 2 \
    --timeout 300 \
    --bind 0.0.0.0:${PORT:-10000} \
    --log-level info \
    app:app"]
