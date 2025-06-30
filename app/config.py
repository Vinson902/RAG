import logging
from pydantic_settings import BaseSettings  # type: ignore

# .env is actual settings, this file is defaults settigns in case .env is inaccessable
class Settings(BaseSettings):
    # Logging settings
    log_level: str = "INFO"
    log_file_path: str = "logs/RAG-app.log"

    # Database
    database_url: str = "postgresql://user:pass@localhost/db"

    # AI Model
    model: str = "Phi-3.5-mini"
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


def setup_logging():  # To be configured!
    """Configure logging with node and pod context"""
    # pod_filter = PodContextFilter()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.DEBUG),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(settings.log_file_path)],
    )
    logging.getLogger("asyncpg").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    # Application loggers
    logging.getLogger("Database.Postgres").setLevel(logging.DEBUG)
    logging.getLogger("LLMClient.LLama").setLevel(logging.INFO)

    # Add filter to add information about pods and nodes
    # logging.getLogger().addFilter(pod_filter)

    logging.info("Logging configuration completed")


# Global settings instance
settings = Settings()
