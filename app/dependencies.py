from typing import Annotated
from fastapi import Depends
from config import settings
from core.database import DatabaseManager,get_database_manager

async def get_database() -> DatabaseManager:
    """Database dependency for FastAPI endpoints"""
    db = await get_database_manager()
    if await db.connect():
        return db
    return None

async def get_llama_client():
    """LLama client dependency"""
    return {"type":"llama_client", "status": "not_implemented"}

async def get_embedding_service():
    """Embedding service dependency"""
    return {"type":"embedding_service", "status": "not_implemented"}

#Aliases
DatabaseDep = Annotated[dict, Depends(get_database)]
LlamaDep = Annotated[dict, Depends(get_llama_client)]
EmbeddingDep = Annotated[dict, Depends(get_embedding_service)]
