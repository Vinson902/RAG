import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional,List, Dict, Any, Tuple
import asyncpg  
from asyncpg import Pool, Connection
from config import settings


class DatabaseManagerBase(ABC):
    """Abstract class for database operations"""

    def __init__(self):
        self.logger = logging.getLogger("Database")
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected
        
    def _set_connected(self, connected: bool):
        """Update connection state with logging"""
        self._connected = connected
        status = "connected" if connected else "disconnected"
        self.logger.info(f"Database {status}")
        
    async def _execute_with_retry(self, operation: Callable, operation_name :str = "", max_retries: int = 3):
        """Retry logic for database operations that commonly raise exceptions"""
        for attempt in range(max_retries):
            try:
                result = await operation()
                self.logger.debug(f"{operation_name} - Succeeded on attempt {attempt + 1}")
                return result
            except Exception as e:

                self.logger.info(f"{operation_name} - Attempt {attempt + 1} caught exception: {type(e).__name__}: {e}")

                if attempt == max_retries - 1:
                    self.logger.error(f"Operation: {operation_name} - failed after {attempt+1} attempts  error: {e}")
                else:
                    self.logger.warning(f"{operation_name} - Attempt {attempt + 1} failed, retrying: {e}")
                    self.logger.debug(f"{operation_name} - Sleeping for {0.1 * (2 ** attempt)} seconds")
                    await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff
        return None


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
        self.logger = logging.getLogger("Database.Postgres")
        self.database_url = database_url
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.pool: Optional[Pool] = None


        self.expected_dims = {          # Known embedding models and their dimensions
        "all-MiniLM-L6-v2": 384,        # Redesign 
        "all-mpnet-base-v2": 768,
        "sentence-t5-base": 768
        }


    async def connect(self) -> None:
        """Initialize connection pool"""
        async def _connect():
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=self.min_connections,
                max_size=self.max_connections,
                command_timeout=60
            )
            # Initialize schema on connection
            await self.init_schema()
            
            # Update state
            self._set_connected(True)
            return True
        
        # If not successful kubernetes should restart the pod
        result = await self._execute_with_retry(_connect, operation_name="database_connection") 
        if result:
            self.logger.info(f"database is connected")        
        return result


    async def disconnect(self) -> None:
        """Close all connections in pool"""
        if self.pool:
            await self.pool.close()
            self._set_connected(False)


    async def health_check(self) -> bool:
        """Check if database is accessible"""
        self.logger.debug("health_check")
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
        
        async with self.pool.acquire() as conn: # Borrow a connection 
            try:
                # Enable pgvector extension 
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                self.logger.info("pgvector extension enabled")
                
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
                
                self.logger.info("Documents table created/verified")

                # Add index for model-based queries
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS documents_embedding_model_idx 
                    ON documents(embedding_model)
                """)
                
                # Create vector similarity index 
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS documents_embedding_idx          # Divides existing entries into clusters 
                    ON documents USING ivfflat (embedding vector_cosine_ops)    # with 100 entries per cluster
                    WITH (lists = 100)                                          # used for quick search
                """)
                
                # Create text search index
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS documents_content_idx 
                    ON documents USING gin(to_tsvector('english', content))
                """)
                
                self.logger.info("Database schema initialization completed")
                
            except Exception as e:
                self.logger.error(f"Schema initialization failed: {e}")
                raise   # Re-raise will make the app handler the exception somewhere else
                        # Or will make kubernetes restart the pod
    
        # Document CRUD operations (to be implemented)
    async def insert_document(self, doc_id: str, content: str, embedding: List[float], 
                            metadata: Dict[str, Any], embedding_model: str) -> bool:
        """Insert document with embedding"""
        
        expected_dim = self.expected_dims.get(embedding_model,384) # all-MiniLM-L6-v2 should be used by default
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
        """Find similar documents using vector search within same embedding model"""

        if not self.pool:
            raise RuntimeError("Database pool not initialized")

        expected_dim = self.expected_dims.get(embedding_model, 384)
        if len(query_embedding) != expected_dim:
            raise ValueError(
                f"Query embedding for model '{embedding_model}' must be {expected_dim}-dimensional, got {len(query_embedding)}"
            )

        # Validate limit parameter
        if limit <= 0 or limit > 100:
            raise ValueError("Limit must be between 1 and 100")

        # Validate embedding values
        if not all(isinstance(x, (int, float)) for x in query_embedding):
            raise ValueError("All embedding values must be numeric")

        # SQL query with pgvector cosine similarity
        query = """
            SELECT 
                id, 
                content, 
                embedding_model, 
                metadata, 
                created_at,
                updated_at,
                embedding <=> $1 AS similarity_distance
            FROM documents 
            WHERE embedding_model = $2
            ORDER BY embedding <=> $1
            LIMIT $3
        """

        async def _search():
            async with self.pool.acquire() as conn:
                # Convert embedding to pgvector format
                embedding_str = f"[{','.join(map(str, query_embedding))}]"

                # Execute similarity search
                rows = await conn.fetch(
                    query,
                    embedding_str,    # $1 - query embedding
                    embedding_model,  # $2 - model filter
                    limit            # $3 - result limit
                )

                # Convert rows to list of dictionaries
                results = []
                for row in rows:
                    result = {
                        "id": row["id"],
                        "content": row["content"],
                        "embedding_model": row["embedding_model"],
                        "metadata": row["metadata"],
                        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
                        "similarity_score": 1.0 - float(row["similarity_distance"])  # Convert distance to similarity
                    }
                    results.append(result)

                return results

        try:
            results = await self._execute_with_retry(_search, operation_name=f"search_similar_{embedding_model}")

            self.logger.debug(
                f"Found {len(results)} similar documents for model {embedding_model} (limit: {limit})"
            )

            return results if results else []

        except ValueError as e:
            # Re-raise validation errors immediately
            self.logger.error(f"Validation error in similarity search: {e}")
            raise

        except Exception as e:
            self.logger.error(f"Failed to execute similarity search: {e}")
            return []



# Global instance for dependency injection
_db_manager: Optional[DatabaseManager] = None

async def get_database_manager() -> DatabaseManager:
    """Factory function - create but don't auto-connect"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(settings.database_url)
        if _db_manager.connect():
            return _db_manager
        return None
    return _db_manager


async def close_database_manager():
    """Cleanup function for application shutdown"""
    global _db_manager
    if _db_manager:
        await _db_manager.disconnect()
        _db_manager = None