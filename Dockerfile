# Use slim Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (git, ffmpeg, and others Whisper needs)
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set env to avoid python buffering issues in logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy requirements and install dependencies (with extra index for torch CPU)
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your app
COPY . .

# Expose the port (important for Azure)
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "model:app", "--host", "0.0.0.0", "--port", "8000"]
