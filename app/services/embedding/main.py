from typing import List
from fastapi import FastAPI, HTTPException, Depends
from contextlib import asynccontextmanager
import logging
import time
import psutil
import uvicorn

from config import settings
from core.embedding_service import EmbeddingService

from core.models import (
    EmbedRequest,
    EmbedBatchRequest,
    EmbedItem,
    HealthResponse
)

# Configure logging
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# Global service instance
start_time = time.time()


# Load the model before the api, so kubernetes could check service availability before labelling it as running
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("Starting embedding service...")
    logger.info(f"Model: {settings.model_name}")
    try:
        global embedding_service
        embedding_service = EmbeddingService(settings.model_name)
        logger.info("Embedding service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize embedding service: {e}")
        exit(1)
    yield
    # Shutdown
    logger.info("Shutting down embedding service...")


# Create FastAPI app
app = FastAPI(
    title="Embedding Service",
    description="Microservice for text embeddings using sentence-transformers",
    version="1.0.0",
    lifespan=lifespan,
)


def get_embedding_service() -> EmbeddingService:
    """Dependency to get embedding service"""
    return embedding_service


@app.post("/embed", response_model=EmbedItem)
async def embed_text(
    request: EmbedRequest, service: EmbeddingService = Depends(get_embedding_service)
):
    """Convert single text to embedding vector"""
    try:
        logger.info(f"Embedding request for text length: {len(request.text)}")
        embedding = service.encode_text(request.text)

        return embedding

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate embedding")


@app.post("/embed/batch", response_model=List[EmbedItem])
async def embed_batch(
    request: EmbedBatchRequest,
    service: EmbeddingService = Depends(get_embedding_service),
):
    """Convert multiple texts to embedding vectors"""
    try:
        logger.info(f"Batch embedding request for {len(request.texts)} texts")

        embeddings = service.encode_batch(request.texts)

        return embeddings

    except ValueError as e:
        logger.error(f"Batch validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch embedding error: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to generate batch embeddings"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check(service: EmbeddingService = Depends(get_embedding_service)):
    """Health check endpoint for k3s"""
    try:
        # Test embedding
        test_embedding = service.encode_text("health check")

        # Get system info
        memory_percent = psutil.virtual_memory().percent
        uptime = time.time() - start_time

        return HealthResponse(
            status="healthy",
            model_loaded=service.is_loaded,
            model_name=service.model_name,
            dimensions=test_embedding.dimensions,
            memory_usage=f"{memory_percent:.1f}%",
            uptime_seconds=uptime,
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.get("/info")
async def get_info(service: EmbeddingService = Depends(get_embedding_service)):
    """Get model information"""
    return service.get_model_info()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=False,  # No reload in production
        workers=1,  # Single worker for Pi
    )
