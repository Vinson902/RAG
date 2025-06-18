-- docker/postgres/init-db.sql
-- RAG Chatbot Database Schema Initialization
-- This script runs automatically on first container startup

-- ============================================================================
-- EXTENSION: Enable pgvector for vector operations
-- ============================================================================
CREATE EXTENSION IF NOT EXISTS vector;


-- ============================================================================
-- TABLE: Main documents storage with vector embeddings
-- ============================================================================
CREATE TABLE IF NOT EXISTS documents (
    -- Primary key: Unique identifier for each document
    id VARCHAR(255) PRIMARY KEY,

    -- Document content: The actual text we'll search through
    content TEXT NOT NULL,

    -- Vector embedding: Numerical representation for similarity search
    embedding vector(384),

    -- Metadata: Additional information about the document
    metadata JSONB,

    -- Timestamp: When document was added to database
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- ============================================================================
-- INDEX: Fast vector similarity search
-- ============================================================================
CREATE INDEX IF NOT EXISTS documents_embedding_idx 
ON documents USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);


-- ============================================================================
-- INDEX: Full-text search on document content
-- ============================================================================
CREATE INDEX IF NOT EXISTS documents_content_idx 
ON documents USING gin(to_tsvector('english', content));

-- ============================================================================
-- INDEX: Fast metadata queries (optional but useful)
-- ============================================================================
CREATE INDEX IF NOT EXISTS documents_metadata_idx 
ON documents USING gin(metadata);


-- ============================================================================
-- VERIFICATION: Check our setup worked
-- ============================================================================
-- Show available extensions
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Show table structure
\d documents;

-- Show indexes created
SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'documents';