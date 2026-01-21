# Use Python 3.9 base image for TensorFlow compatibility
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt update -y && apt install awscli -y

# Set working directory
WORKDIR /app

COPY . /app

# Install Python dependencies
RUN pip install -r requirements.txt


#   - App: python app.py
CMD ["python2", "app.py"]

