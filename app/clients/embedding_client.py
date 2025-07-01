import httpx
import asyncio
import logging
from typing import Optional, List, ClassVar

from pydantic import BaseModel, Field

from client import Client
from config import settings

# Copied from embedding service models module
# To be changed
class EmbedRequest(BaseModel):
    text: str = Field(
        description="Text to embed",
        min_length=1,
        max_length=10000
    )


class EmbedBatchRequest(BaseModel):
    texts: List[str] = Field(
        description="List of texts to embed",
        min_length=1,
        max_length=100
    )

 # Responses

class EmbedItem(BaseModel):
    text: str = Field(description="Original text")
    embedding: List[float] = Field(description="Text embedding vector")
    dimensions: int = Field(description="Number of dimensions")
    text_length: int = Field(description="Original text length")
    model: str = Field(description="Model used for embeddings")

class HealthResponse(BaseModel):
    status: str = Field(description="Health status")
    model_loaded: bool = Field(description="Whether model is loaded")
    model_name: str = Field(description="Model name")
    dimensions: Optional[int] = Field(default=None, description="Model dimensions")
    memory_usage: str = Field(description="Memory usage percentage")
    uptime_seconds: float = Field(description="Service uptime")

class ErrorResponse(BaseModel):
    error: str = Field(description="Error message")
    detail: Optional[str] = Field(default=None, description="Error details")

class ModelInfoResponse(BaseModel):
    model_name: str = Field(description="Model name")
    is_loaded: bool = Field(description="Whether model is loaded")
    dimensions: Optional[int] = Field(default=None, description="Model dimensions")

class EmbeddingClient(Client):

    def __init__(
        self, 
        base_url: str, 
        timeout: float = 30.0, 
        max_retries: int = 3
    ):
        """
        Initialize embedding client.
        
        Args:
            service_url: URL of embedding service
            timeout: Request timeout in seconds (default: 30.0)
            max_retries: Maximum retry attempts (default: 3)
        """
        super().__init__(base_url, timeout, max_retries)

    async def health_check(self) -> str:
        try:
            result = await self.client.get("/health")
            response = HealthResponse.model_validate_json(result.text)
            return response.status
        except Exception as e:
            self.logger.warning(f"Health check failed {e}")
            raise

    async def embed_text(self,text: str):


        return await self.client.post("/embed", json=EmbedRequest(
            text= text
        ))
    def embed_batch(self, texts: List[str]):
        return

