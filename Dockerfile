# Start from a maintained slim image
FROM python:3.11-slim

# Environment for leaner, safer Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install only what you need; clean apt caches
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# (Optional) If you truly need AWS CLI inside this image:
# - apt often installs awscli v1; v2 is usually installed via official script.
# If v1 is fine:
# RUN apt-get update && apt-get install -y --no-install-recommends awscli && rm -rf /var/lib/apt/lists/*
# If you want AWS CLI v2, consider running a separate `amazon/aws-cli` container instead of baking it in.

# Create app directory and user
WORKDIR /app
RUN useradd -m appuser

# Install Python deps first (better Docker cache)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the source
COPY --chown=appuser:appuser . /app

# Drop root privileges
USER appuser

# If this is a web app, consider gunicorn instead of python
# EXPOSE 8000
# CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]

# For a simple script entrypoint:
CMD ["python", "app.py"]
