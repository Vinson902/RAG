FROM python:3.11-slim

# Set environment variables to limit model downloads
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
ENV TRANSFORMERS_OFFLINE=0
ENV HF_DATASETS_OFFLINE=0

# Handpicked libraries to minimase the image size 
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    numpy==1.24.4 \
    transformers==4.35.0 \
    sentence-transformers==2.2.2 \
    huggingface-hub==0.17.3 \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    psutil \
    numpy \
    pydantic_settings \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

# Create app directory and copy files
WORKDIR /app
COPY services/embedding/main.py .
COPY services/embedding/config.py .
COPY services/embedding/core/ ./core/

# Expose port
EXPOSE 8001

# Run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]