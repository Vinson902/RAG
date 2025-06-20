from typing import Annotated
from fastapi import Depends
from config import settings
from core.database import DatabaseManager, get_database_manager
from core.llama_client import LlamaClient, create_llama_client 


async def get_database() -> DatabaseManager:
    """Database dependency for FastAPI endpoints"""
    db = await get_database_manager()
    return db

async def get_llama_client():
    """LLama client dependency"""
    client = create_llama_client()
    try:
        yield client
    finally:
        client.close()

async def get_embedding_service():
    """Embedding service dependency"""
    return {"type":"embedding_service", "status": "not_implemented"}

#Aliases
DatabaseDep = Annotated[DatabaseManager, Depends(get_database)]
LlamaDep = Annotated[LlamaClient, Depends(get_llama_client)]
EmbeddingDep = Annotated[dict, Depends(get_embedding_service)]
