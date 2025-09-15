# Dockerfile
FROM python:3.11-slim

# Needed by scikit-learn 
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install deps first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src ./src

# copy docs/Makefile/etc if you want
COPY Makefile ./

# We'll mount data/models at runtime, so we don't bake them into the image.
EXPOSE 8080

# Default CMD runs the API
CMD ["uvicorn", "--app-dir", "src", "API.fast_api:app", "--host", "0.0.0.0", "--port", "8080"]

