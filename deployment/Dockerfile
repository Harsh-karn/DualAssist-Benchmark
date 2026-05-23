FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements_spaces.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app_spaces.py app.py

# Expose Gradio default port
EXPOSE 7860

# Set environment
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Run
CMD ["python", "app.py"]
