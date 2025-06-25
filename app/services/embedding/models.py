from pydantic import BaseModel, Field
from typing import List, Optional

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

class EmbedResponse(BaseModel):
    embedding: List[float] = Field(description="Text embedding vector")
    dimensions: int = Field(description="Number of dimensions")
    text_length: int = Field(description="Original text length")

class EmbedBatchResponse(BaseModel):
    embeddings: List[List[float]] = Field(description="List of embedding vectors")
    dimensions: int = Field(description="Number of dimensions")
    count: int = Field(description="Number of embeddings returned")

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

class ModelInfo(BaseModel):
    model_name: str = Field(description="Model name")
    is_loaded: bool = Field(description="Whether model is loaded")
    dimensions: Optional[int] = Field(default=None, description="Model dimensions")