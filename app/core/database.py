import asyncpg
import asyncio
import logging
import json
from typing import List, Dict, Any, Optional, ClassVar
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

    def __init__(self):
        self.logger = logging.getLogger("DatabaseClient")

    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create shared connection pool"""
        if self._pool is None or self._pool._closed:
            async with self._pool_lock:
                # Double-check pattern
                if self._pool is None or self._pool._closed:
                    dbc = settings.database_url
                    # Database connection string
                    dbc
                    self._pool = await asyncpg.create_pool(
                        dbc,
                        min_size=2,  # Minimum connections
                        max_size=10,  # Maximum connections
                        max_queries=50000,  # Max queries per connection
                        max_inactive_connection_lifetime=300,  # 5 minutes
                        command_timeout=60,  # Command timeout
                        server_settings={
                            "jit": "off",  # Disable JIT for Pi performance
                            "application_name": "pi-cluster-rag",
                        },
                    )

                    # Initialize database schema
                    await self._initialize_schema()

                    self.logger.info("Created database connection pool")

        return self._pool

    async def _initialize_schema(self):
        """Initialize database schema and pgvector extension"""
        async with self._pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Create documents table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding VECTOR(384),  
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)

            # Create index for vector similarity search
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS documents_embedding_idx 
                ON documents USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)

            # Create text search index
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS documents_content_idx 
                ON documents USING GIN (to_tsvector('english', content));
            """)

            self.logger.info("Database schema initialized")

    async def insert_document(
        self, content: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Insert a document with its embedding"""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            # Convert embedding to string format for pgvector
            embedding_str = f"[{','.join(map(str, embedding))}]"

            result = await conn.fetchrow(
                """
                INSERT INTO documents (content, embedding, metadata)
                VALUES ($1, $2, $3)
                RETURNING id;
                """,
                content,
                embedding_str,
                json.dumps(metadata or {}),
            )

            doc_id = result["id"]
            self.logger.debug(f"Inserted document {doc_id}")
            return doc_id

    async def insert_documents_batch(self, documents: List[Dict]) -> List[int]:
        """Insert multiple documents efficiently"""
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

    async def update_document(
        self,
        doc_id: int,
        content: str = "",
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update a document"""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            # Build dynamic update query
            updates = []
            values = []
            param_count = 1

            if content is not None:
                updates.append(f"content = ${param_count}")
                values.append(content)
                param_count += 1

            if embedding is not None:
                embedding_str = f"[{','.join(map(str, embedding))}]"
                updates.append(f"embedding = ${param_count}")
                values.append(embedding_str)
                param_count += 1

            if metadata is not None:
                updates.append(f"metadata = ${param_count}")
                values.append(json.dumps(metadata))
                param_count += 1

            if not updates:
                return False

            updates.append("updated_at = NOW()")
            values.append(doc_id)  # For WHERE clause

            query = f"""
                UPDATE documents 
                SET {", ".join(updates)}
                WHERE id = ${param_count};
            """

            result = await conn.execute(query, *values)
            updated = result.split()[-1] == "1"

            if updated:
                self.logger.debug(f"Updated document {doc_id}")

            return updated

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
