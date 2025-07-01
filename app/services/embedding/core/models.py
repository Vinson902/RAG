from pydantic import BaseModel, Field
from typing import List, Optional


 # Requests 
 
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