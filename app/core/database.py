import logging
from abc import ABC, abstractmethod
from typing import Optional,List, Dict, Any, Tuple
import asyncpg
from asyncpg import Pool, Connection
from config import settings

logger = logging.getLogger(__name__)

class DatabaseManagerBase(ABC):
    """Abstract class for database operations"""
    @abstractmethod
    async def connect(self):
        """initialise database connection"""
        pass
    @abstractmethod
    async def disconnect(self):
        """Close connection with the database"""
        pass
    @abstractmethod
    async def health_check(self):
        """Connectivity  check"""
        pass

class DatabaseManager(DatabaseManagerBase):
    """
    Manages PostgreSQL + pgvector operations for document storage, retrieval and search
    
    Responsibilities:
    - Connection pool management
    - Schema initialization 
    - Document CRUD operations
    - Vector operations
    """
    def __init__(self, database_url: str, min_connections: int = 2, max_connections: int = 5):
        self.database_url = database_url
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.pool: Optional[Pool] = None
        self._initialized = False


    async def connect(self) -> None:
        """Initialize connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=self.min_connections,
                max_size=self.max_connections,
                command_timeout=30                  # the host should be local or via ethernet
            )
            logger.info(f"Connected to database with pool size {self.min_connections}-{self.max_connections}")
            
            # Initialize schema on connection
            await self.init_schema()
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise


    async def disconnect(self) -> None:
        """Close all connections in pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connections closed")


    async def health_check(self) -> bool:
        """Check if database is accessible"""
        if not self.pool:
            return False
            
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    

    async def init_schema(self) -> None:
        """Initialize database schema with documents table and vector index"""
        if not self.pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self.pool.acquire() as conn:
            try:
                # Enable pgvector extension
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                logger.info("pgvector extension enabled")
                
                # Create embedding table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id VARCHAR(255) PRIMARY KEY,
                        content TEXT NOT NULL,
                        embedding vector(384),
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                logger.info("Documents table created/verified")
                
                # Create vector similarity index
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS documents_embedding_idx 
                    ON documents USING ivfflat (embedding vector_cosine_ops) 
                    WITH (lists = 100)
                """)
                
                # Create text search index
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS documents_content_idx 
                    ON documents USING gin(to_tsvector('english', content))
                """)
                
                # Create metadata index
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS documents_metadata_idx 
                    ON documents USING gin(metadata)
                """)
                
                logger.info("Database schema initialization completed")
                
            except Exception as e:
                logger.error(f"Schema initialization failed: {e}")
                raise
    
        # Document CRUD operations (to be implemented)
    async def insert_document(self, doc_id: str, content: str, embedding: List[float], 
                            metadata: Dict[str, Any]) -> bool:
        """Insert document with embedding"""
        pass
    
    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document by ID"""
        pass
    
    async def search_similar(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar documents using vector search"""
        pass

# Global instance for dependency injection
_db_manager: Optional[DatabaseManager] = None

async def get_database_manager() -> DatabaseManager:
    """
    Factory function for creating/returning DatabaseManager instance
    To be used in your dependencies.py get_database() function
    """
    global _db_manager
    if _db_manager is None:
        # This will be imported from config.py
        _db_manager = DatabaseManager(settings.database_url)
        await _db_manager.connect()
    return _db_manager

async def close_database_manager():
    """Cleanup function for application shutdown"""
    global _db_manager
    if _db_manager:
        await _db_manager.disconnect()
        _db_manager = None