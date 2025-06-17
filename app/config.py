import os
from typing import Optional
from pydantic_settings import BaseSettings # type: ignore


#.env is actual settings, this file is defaults settigns in case .env is inaccessable 
class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://user:pass@localhost/db"
    
    # AI Model
    llama_host: str = "localhost"
    llama_port: int = 8080
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # App
    app_name: str = "RAG Chatbot"
    app_version: str = "1.0.0"

    # CORS
    cors_origins: str = "*"
    cors_allow_credentials: bool = True 
    
    class Config:
        env_file = ".env"

# Global settings instance
settings = Settings()