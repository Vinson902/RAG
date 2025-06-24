# main.py - Clean integration with both LLM and Database
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from typing import List
from core.llama_client import LlamaClient
from core.database import DatabaseClient

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    print("🚀 Starting Pi-Cluster RAG application...")
    
    # Test connections on startup
    llama = LlamaClient()
    db = DatabaseClient()
    
    if not await llama.health_check():
        raise RuntimeError("❌ LLM server not available")
    
    if not await db.health_check():
        raise RuntimeError("❌ Database not available")
    
    print("All services ready!")
    
    yield  # Application runs
    
    # Cleanup on shutdown
    await LlamaClient.close_shared_client()
    await DatabaseClient.close_pool()
    print("Application shutdown complete")

app = FastAPI(
    title="Pi-Cluster RAG API",
    description="RAG system with vector search",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Check all services health"""
    llama = LlamaClient()
    db = DatabaseClient()
    
    llm_healthy = await llama.health_check()
    db_healthy = await db.health_check()
    
    return {
        "status": "healthy" if (llm_healthy and db_healthy) else "unhealthy",
        "services": {
            "llm": "available" if llm_healthy else "unavailable",
            "database": "available" if db_healthy else "unavailable"
        }
    }

@app.post("/documents")
async def add_document(content: str, metadata: dict = None):
    """Add a document to the knowledge base"""
    # You'll need an embedding service here
    # For now, using dummy embedding
    embedding = [0.1] * 384  # Replace with actual embedding generation
    
    db = DatabaseClient()
    doc_id = await db.insert_document(content, embedding, metadata)
    
    return {"document_id": doc_id, "message": "Document added successfully"}

@app.get("/documents/{doc_id}")
async def get_document(doc_id: int):
    """Get a specific document"""
    db = DatabaseClient()
    document = await db.get_document(doc_id)
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "id": document.id,
        "content": document.content,
        "metadata": document.metadata
    }

@app.post("/search")
async def search_documents(query: str, limit: int = 5):
    """Search documents using text search"""
    db = DatabaseClient()
    documents = await db.search_text(query, limit)
    
    return {
        "query": query,
        "results": [
            {
                "id": doc.id,
                "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                "metadata": doc.metadata,
                "relevance_score": doc.similarity
            }
            for doc in documents
        ]
    }

@app.post("/search/similar")
async def search_similar_documents(query_embedding: List[float], limit: int = 5):
    """Search documents using vector similarity"""
    db = DatabaseClient()
    documents = await db.search_similar(query_embedding, limit)
    
    return {
        "results": [
            {
                "id": doc.id,
                "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                "metadata": doc.metadata,
                "similarity": doc.similarity
            }
            for doc in documents
        ]
    }

@app.post("/rag/chat")
async def rag_chat(question: str, max_context_docs: int = 3):
    """RAG-enabled chat using retrieved context"""
    
    # 1. Search for relevant documents
    db = DatabaseClient()
    relevant_docs = await db.search_text(question, max_context_docs)
    
    if not relevant_docs:
        # No context found, direct LLM response
        llama = LlamaClient()
        response = await llama.generate(
            prompt=question,
            system_message="You are a helpful AI assistant. Answer based on your knowledge."
        )
        
        return {
            "answer": response.content,
            "context_used": False,
            "sources": []
        }
    
    # 2. Build context from retrieved documents
    context = "\n\n".join([
        f"Source {i+1}: {doc.content}"
        for i, doc in enumerate(relevant_docs)
    ])
    
    # 3. Generate response with context
    llama = LlamaClient()
    
    system_message = """You are a helpful AI assistant. Answer the user's question based on the provided context. 
If the context doesn't contain enough information to answer the question, say so clearly.
Be concise and cite which sources you used."""

    prompt = f"""Context:
{context}

Question: {question}

Answer:"""

    response = await llama.generate(
        prompt=prompt,
        system_message=system_message,
        max_tokens=300
    )
    
    if response.error:
        raise HTTPException(status_code=500, detail=f"Generation failed: {response.error}")
    
    return {
        "answer": response.content,
        "context_used": True,
        "sources": [
            {
                "id": doc.id,
                "content_preview": doc.content[:100] + "..." if len(doc.content) > 100 else doc.content,
                "relevance_score": doc.similarity
            }
            for doc in relevant_docs
        ],
        "generation_stats": {
            "tokens_generated": response.tokens_generated,
            "tokens_per_second": response.tokens_per_second
        }
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    db = DatabaseClient()
    db_stats = await db.get_stats()
    
    return {
        "database": db_stats,
        "system": {
            "version": "1.0.0",
            "services": ["llm", "database", "vector_search"]
        }
    }

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: int):
    """Delete a document"""
    db = DatabaseClient()
    success = await db.delete_document(doc_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"message": f"Document {doc_id} deleted successfully"}

# Simple generation endpoint (non-RAG)
@app.post("/generate")
async def generate_text(prompt: str, max_tokens: int = 200):
    """Direct text generation without RAG"""
    llama = LlamaClient()
    
    if not await llama.health_check():
        raise HTTPException(status_code=503, detail="LLM service unavailable")
    
    response = await llama.generate(prompt=prompt, max_tokens=max_tokens)
    
    if response.error:
        raise HTTPException(status_code=500, detail=f"Generation failed: {response.error}")
    
    return {
        "response": response.content,
        "metadata": {
            "tokens_generated": response.tokens_generated,
            "tokens_per_second": response.tokens_per_second
        }
    }