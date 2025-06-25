import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Service settings
    service_name: str = "embedding-service"
    host: str = "0.0.0.0"
    port: int = 8001
    
    # Model settings - MiniLM as default
    model_name: str = "all-MiniLM-L6-v2"  # 384 dimensions, ~80MB
    max_batch_size: int = 32
    
    # Performance settings for Pi
    max_workers: int = 1
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_prefix = "EMBED_"

settings = Settings()