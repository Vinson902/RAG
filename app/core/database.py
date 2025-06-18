import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional,List, Dict, Any, Tuple
import asyncpg
from asyncpg import Pool, Connection
from config import settings

logger = logging.getLogger(__name__)

class DatabaseManagerBase(ABC):
    """Abstract class for database operations"""

    def __init__(self):
        self.logger = logging.getLogger("core.database")
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected
        
    def _set_connected(self, connected: bool, url: str = None):
        """Update connection state with logging"""
        self._connected = connected
        if url:
            self._connection_url = url
        status = "connected" if connected else "disconnected"
        self.logger.info(f"Database {status}")
        
    async def _execute_with_retry(self, operation: Callable, operation_name :str = "", max_retries: int = 3):
        """Common retry logic for database operations"""
        for attempt in range(max_retries):
            try:
                result = await operation()
                self.logger.debug(f"{operation_name} - Succeeded on attempt {attempt + 1}")
                return result
            except Exception as e:

                self.logger.debug(f"{operation_name} - Attempt {attempt + 1} caught exception: {type(e).__name__}: {e}")

                if attempt == max_retries - 1:
                    self.logger.error(f"Operation: {operation_name} - failed after {attempt+1} attempts  error: {e}")
                    #raise
                
                self.logger.warning(f"{operation_name} - Attempt {attempt + 1} failed, retrying: {e}")
                self.logger.debug(f"{operation_name} - Sleeping for {0.1 * (2 ** attempt)} seconds")
                await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff


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
        super().__init__()
        self.database_url = database_url
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.pool: Optional[Pool] = None


    async def connect(self) -> None:
        """Initialize connection pool"""
        async def _connect():
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=self.min_connections,
                max_size=self.max_connections,
                command_timeout=60
            )
            # Use base class method to update state
            self._set_connected(True, self.database_url)
            
            # Initialize schema on connection
            await self.init_schema()
            return True
        
        result = await self._execute_with_retry(_connect, operation_name="database_connection")
        logging.info("database is connected")
        return result


    async def disconnect(self) -> None:
        """Close all connections in pool"""
        if self.pool:
            await self.pool.close()
            self._set_connected(False)


    async def health_check(self) -> bool:
        """Check if database is accessible"""
        logger.debug("health_check")
        if not self.pool:
            return False
        async def _health_check():
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        
        try:
            # Use base class retry for health checks
            return await self._execute_with_retry(_health_check, max_retries=2, operation_name="health_check")
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
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
                        embedding_model VARCHAR(100) NOT NULL,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                logger.info("Documents table created/verified")

                # Add index for model-based queries
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS documents_embedding_model_idx 
                    ON documents(embedding_model)
                """)
                
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
                            metadata: Dict[str, Any], embedding_model: str) -> bool:
        """Insert document with embedding"""
        
        ###
        ### Redesign 
        ###

        expected_dims = {           # Known embedding models and their dimensions
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "sentence-t5-base": 768
        }


        expected_dim = expected_dims.get(embedding_model,384) # all-MiniLM-L6-v2 should be used by default
        # Validate embedding dimensions
        if len(embedding) != expected_dim:
            raise ValueError(f"Embedding for {embedding_model} should be {expected_dim}, got {len(embedding)}")

        # Ensure metadata is not None for JSONB storage
        if metadata is None:
            metadata = {}

        # SQL query with proper pgvector syntax
        query = """
            INSERT INTO documents (id, content, embedding, embedding_model, metadata, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, NOW(), NOW())       #To avoid sql-injection  
            ON CONFLICT (id) 
            DO UPDATE SET 
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                embedding_model = EXCLUDED.embedding_model,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
        """

        async def _insert():
            async with self.pool.acquire() as conn:
                # Convert embedding list to pgvector format
                embedding_str = f"[{','.join(map(str, embedding))}]"    # transform array into a string as pgvector expects 
                                                                        # a string value with vectors 
                # Execute the insert/update query
                await conn.execute(
                    query,
                    doc_id,           # $1
                    content,          # $2
                    embedding_str,    # $3 - pgvector format
                    embedding_model,  # $4 - embedding model
                    metadata          # $5 - JSONB format (asyncpg transforms PostrgreSQL types to python types)
                )
                return True

        try:
            result = await self._execute_with_retry(_insert, operation_name=f"insert_document_{doc_id}")
            if result:
                self.logger.info(f"Successfully inserted/updated document: {doc_id} with model {embedding_model}")
            return result if result else False

        except ValueError as e:
            # Validation errors
            self.logger.error(f"Validation error for document {doc_id}: {e}")
            raise

        except Exception as e:
            self.logger.error(f"Failed to insert document {doc_id}: {e}")
            return False
    

    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document by ID"""

        if not self.pool:
            raise RuntimeError("Database pool not initialized")

        # SQL query to fetch document by ID
        query = """
            SELECT id, content, embedding, embedding_model, metadata, created_at, updated_at
            FROM documents 
            WHERE id = $1
        """

        async def _get():
            async with self.pool.acquire() as conn:
                # Execute query and fetch one row
                row = await conn.fetchrow(query, doc_id)

                if row is None:
                    return None
                
                # Convert asyncpg.Record to dictionary
                result = {
                    "id": row["id"],
                    "content": row["content"],
                    "embedding_model": row["embedding_model"],
                    # Include embedding vector if needed
                    #"embedding": result["embedding"] = list(row["embedding"]) if row["embedding"] else None 
                    "metadata": row["metadata"],  # asyncpg auto-converts JSONB to dict
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None
                }

                return result

        try:
            result = await self._execute_with_retry(_get, operation_name=f"get_document_{doc_id}")

            if result:
                self.logger.debug(f"Successfully retrieved document: {doc_id}")
            else:
                self.logger.debug(f"Document not found: {doc_id}")

            return result

        except Exception as e:
            self.logger.error(f"Failed to retrieve document {doc_id}: {e}")
            return None

    async def search_similar(self, query_embedding: List[float], embedding_model: str, 
                            limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar documents using vector search - ONLY within the same embedding model using cosine distance"""

        query = """
            SELECT id, content, metadata, embedding_model,
                   embedding <=> %s AS similarity_distance 
            FROM documents 
            WHERE embedding_model = %s
            ORDER BY embedding <=> %s
            LIMIT %s
        """



# Global instance for dependency injection
_db_manager: Optional[DatabaseManager] = None

async def get_database_manager() -> DatabaseManager:
    """Factory function - create but don't auto-connect"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(settings.database_url)
        logging.debug("Created DatabaseManager instance (not connected)")
    return _db_manager


async def close_database_manager():
    """Cleanup function for application shutdown"""
    global _db_manager
    if _db_manager:
        await _db_manager.disconnect()
        _db_manager = None