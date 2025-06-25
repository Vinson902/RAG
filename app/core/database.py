import asyncpg
import asyncio
import logging
import json
from typing import Callable, List, Dict, Any, Optional, ClassVar
from dataclasses import dataclass
from config import settings


@dataclass
class Document:
    """Document with embedding for vector search"""

    id: int
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    similarity: Optional[float] = None

class DatabaseClient:
    """
    Database client with internal connection pool management
    Handles PostgreSQL + pgvector operations
    """

    # Class-level shared connection pool
    _pool: ClassVar[Optional[asyncpg.Pool]] = None
    _pool_lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _logger: ClassVar[logging.Logger] = logging.getLogger("DatabaseClient")

    def __init__(self):
        self.logger = self._logger

    @classmethod
    async def _init_pool(cls) -> asyncpg.Pool:
        """
        Initialize the shared connection pool
        This is called automatically when needed
        """
        if cls._pool is None or cls._pool._closed:
            async with cls._pool_lock:
                # Double-check pattern: check again inside lock
                if cls._pool is None or cls._pool._closed:
                    
                    # Get database connection string from settings
                    db_url = settings.database_url
                    
                    cls._logger.info(f"Creating database connection pool to {db_url}")
                    
                    cls._pool = await asyncpg.create_pool(
                        db_url,
                        min_size=2,          # Minimum connections in pool
                        max_size=10,         # Maximum connections in pool  
                        max_queries=50000,   # Max queries per connection before rotation
                        max_inactive_connection_lifetime=300,  # 5 minutes
                        command_timeout=60,  # Individual command timeout
                        server_settings={
                            'jit': 'off',  # Disable JIT compilation for Pi performance
                            'application_name': 'pi-cluster-rag'
                        }
                    )
                    
                    # Initialize database schema
                    await cls._init_schema()
                    
                    cls._logger.info("Database connection pool created successfully")
        
        return cls._pool

    @classmethod
    async def _init_schema(cls):
        """Initialize database schema and extensions"""
        async with cls._pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create documents table if it doesn't exist
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding VECTOR(384),  -- Adjust size based on your embedding model
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)
            
            # Create vector similarity index
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS documents_embedding_cosine_idx 
                ON documents USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            
            # Create text search index
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS documents_content_search_idx 
                ON documents USING GIN (to_tsvector('english', content));
            """)
            
            # Create metadata index for filtering
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS documents_metadata_idx 
                ON documents USING GIN (metadata);
            """)
            
            cls._logger.info("Database schema initialized")

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
    
    async def _get_pool(self) -> asyncpg.Pool:
        """
        Get the shared connection pool
        Initializes pool if it doesn't exist
        """
        if self._pool is None or self._pool._closed:
            await self._init_pool()
            
        return self._pool
    
    async def insert_document(
        self, content: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Insert a document with its embedding"""

        async def _insert():
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                # Convert embedding to string format for pgvector
                embedding_str = f"[{','.join(map(str, embedding))}]"

                return await conn.fetchrow(
                    """
                    INSERT INTO documents (content, embedding, metadata)
                    VALUES ($1, $2, $3)
                    RETURNING id;
                    """,
                    content,
                    embedding_str,
                    json.dumps(metadata or {}),
                )
            

            result = self._execute_with_retry(_insert, "Insert document")
            doc_id = result["id"]
            self.logger.debug(f"Inserted document {doc_id}")
            return doc_id

    async def insert_documents_batch(self, documents: List[Dict]) -> List[int]:
        """Insert multiple documents efficiently"""

        async def _insert_batch():
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                # Prepare data for batch insert
                data = []
                for doc in documents:
                    embedding_str = f"[{','.join(map(str, doc['embedding']))}]"
                    data.append(
                        (doc["content"], embedding_str, json.dumps(doc.get("metadata", {})))
                    )

                # Batch insert
                result = await conn.fetch(
                    """
                    INSERT INTO documents (content, embedding, metadata)
                    SELECT * FROM UNNEST($1::text[], $2::vector[], $3::jsonb[])
                    RETURNING id;
                    """,
                    [d[0] for d in data],  # content
                    [d[1] for d in data],  # embeddings
                    [d[2] for d in data],  # metadata
                )


            result = self._execute_with_retry(_insert_batch,"insert batch")
            doc_ids = [row["id"] for row in result]
            self.logger.info(f"Inserted {len(doc_ids)} documents")
            return doc_ids

    async def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 5,
        similarity_threshold: float = 0.7,
    ) -> List[Document]:
        """Search for similar documents using vector similarity"""

        async def _search():
            pool = await self._get_pool()

            async with pool.acquire() as conn:
                embedding_str = f"[{','.join(map(str, query_embedding))}]"

                result = await conn.fetch(
                    """
                    SELECT 
                        id, 
                        content, 
                        embedding,
                        metadata,
                        1 - (embedding <=> $1::vector) AS similarity
                    FROM documents
                    WHERE 1 - (embedding <=> $1::vector) > $2
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3;
                    """,
                    embedding_str,
                    similarity_threshold,
                    limit,
                )
            
            resutl = self._execute_with_retry(_search, "Search")
            documents = []
            for row in result:
                # Parse embedding back to list
                embedding = [float(x) for x in row["embedding"].strip("[]").split(",")]

                documents.append(
                    Document(
                        id=row["id"],
                        content=row["content"],
                        embedding=embedding,
                        metadata=json.loads(row["metadata"]),
                        similarity=float(row["similarity"]),
                    )
                )

            self.logger.debug(f"Found {len(documents)} similar documents")
            return documents

    async def search_text(self, query: str, limit: Optional[int] = 10) -> List[Document]:
        """Search documents using full-text search"""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.fetch(
                """
                SELECT 
                    id, 
                    content, 
                    embedding,
                    metadata,
                    ts_rank(to_tsvector('english', content), plainto_tsquery('english', $1)) as rank
                FROM documents
                WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $1)
                ORDER BY rank DESC
                LIMIT $2;
                """,
                query,
                limit,
            )

            documents = []
            for row in result:
                # Parse embedding
                embedding = [float(x) for x in row["embedding"].strip("[]").split(",")]

                documents.append(
                    Document(
                        id=row["id"],
                        content=row["content"],
                        embedding=embedding,
                        metadata=json.loads(row["metadata"]),
                        similarity=float(row["rank"]),  # Using rank as similarity score
                    )
                )

            self.logger.debug(f"Found {len(documents)} documents for text search")
            return documents

    async def get_document(self, doc_id: int) -> Optional[Document]:
        """Get a specific document by ID"""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.fetchrow(
                "SELECT id, content, embedding, metadata FROM documents WHERE id = $1;",
                doc_id,
            )

            if not result:
                return None

            # Parse embedding
            embedding = [float(x) for x in result["embedding"].strip("[]").split(",")]

            return Document(
                id=result["id"],
                content=result["content"],
                embedding=embedding,
                metadata=json.loads(result["metadata"]),
            )

    async def delete_document(self, doc_id: int) -> bool:
        """Delete a document"""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.execute("DELETE FROM documents WHERE id = $1;", doc_id)

            deleted = result.split()[-1] == "1"  # "DELETE 1" means success
            if deleted:
                self.logger.debug(f"Deleted document {doc_id}")

            return deleted


    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.fetchrow(
                """
                SELECT 
                    COUNT(*) as total_documents,
                    AVG(LENGTH(content)) as avg_content_length,
                    MAX(created_at) as latest_document,
                    MIN(created_at) as earliest_document
                FROM documents;
                """
            )

            return {
                "total_documents": result["total_documents"],
                "avg_content_length": float(result["avg_content_length"] or 0),
                "latest_document": result["latest_document"],
                "earliest_document": result["earliest_document"],
            }

    async def health_check(self) -> bool:
        """Check database connectivity"""
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1;")
            return True
        except Exception as e:
            self.logger.warning(f"Database health check failed: {e}")
            return False

    @classmethod
    async def close_pool(cls):
        """Close the shared connection pool (call during app shutdown)"""
        if cls._pool and not cls._pool._closed:
            await cls._pool.close()
            cls._pool = None
            logging.getLogger("DatabaseClient").info("Closed database connection pool")
