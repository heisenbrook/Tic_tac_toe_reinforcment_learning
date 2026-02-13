FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System basics only
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Install required Python packages
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy code
COPY src ./src
COPY templates ./templates
COPY app.py ./app.py
COPY main.py ./main.py

# Standard dirs
RUN mkdir -p artifacts logs

EXPOSE 5000

# Default: Serve the Flask UI
CMD ["python", "main.py", "serve"]