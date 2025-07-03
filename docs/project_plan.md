# Pi-Cluster RAG Chatbot - Project Plan

## 🎯 Project Overview

**Build a distributed RAG (Retrieval-Augmented Generation) chatbot system running on a Raspberry Pi cluster that demonstrates edge AI capabilities on consumer hardware.**

### Value of this project 
- **Edge AI Implementation** - Complete AI system running locally without cloud dependency
- **Distributed Architecture** - Microservices on Kubernetes (k3s) with resource-based optimization
- **Production-Ready Stack** - FastAPI, PostgreSQL+pgvector, Docker, monitoring
- **Hardware Constraints** - Real-world optimization for ARM64 and limited RAM
- **Full MLOps Pipeline** - From development to deployment with CI/CD

---

## 🏗️ System Architecture

### Hardware Configuration
```
pi-master (Raspberry Pi 5):
├── 8GB RAM, SSD with shared data partition
├── IP: 10.0.0.10
├── OS: Ubuntu 24.04.2 Server
└── Role: Database, main application, embeddings

pi-worker (Raspberry Pi 5):
├── 4GB RAM
├── IP: 10.0.0.11  
├── OS: Ubuntu 24.04.2 Server
└── Role: LLM serving, document processing
```

### Microservices Architecture (Resource-Based Split)
```
Pod 1: Main Application (pi-master)
├── FastAPI Gateway & API endpoints
├── Database Manager (PostgreSQL+pgvector)
├── Vector Search operations
├── RAG Pipeline orchestration
└── Memory: ~5-6GB

Pod 2: Embedding Service (pi-master)
├── sentence-transformers model (all-MiniLM-L6-v2)
├── Text-to-vector conversion (384 dimensions)
├── Batch processing optimization
└── Memory: ~600-800MB (isolated)

Pod 3: Document & LLM Client (pi-worker)
├── Document processor (PDF, DOCX, text chunking)
├── LLaMA.cpp HTTP client
├── Co-located with Phi-3.5-mini Q3_K_M model
└── Memory: ~2GB
```

### Technology Stack
- **AI/ML:** llama.cpp with Phi-3.5-mini Q3_K_M, sentence-transformers, torch
- **Backend:** Python FastAPI, asyncio, xhttp, uvicorn
- **Database:** PostgreSQL + pgvector extension, asyncpg
- **Document Processing:** pypdf, python-docx, beautifulsoup4, lxml
- **Orchestration:** k3s (lightweight Kubernetes), Docker
- **Monitoring:** Prometheus, Grafana (external)
- **Infrastructure:** ARM64 optimization, resource-constrained deployment

---

## 📋 Development Phases

### Phase 1: System Preparation
**Objective:** Set up Raspberry Pi cluster with k3s orchestration
- Install essential packages on both Pis (Docker, Git, build tools)
- Set up nfs server on pi-master so share a partition of SSD
- Deploy k3s server on pi-master (control plane)
- Join pi-worker as k3s agent node
- Verify cluster connectivity and node status
- Configure shared storage between nodes

### Phase 2: Model Setup (pi-worker)
**Objective:** Deploy and optimize LLM serving infrastructure
- Clone and compile llama.cpp with ARM64 optimizations
- Download Phi-3.5-mini Q3_K_M model (optimized for 4GB RAM)
- Configure model server with memory-aware settings
- Test inference and optimize for performance
- Set up REST API endpoint for model serving

### Phase 3: Database Setup (pi-master)
**Objective:** Implement vector database for semantic search
- Install PostgreSQL with development dependencies
- Build and install pgvector extension for vector operations
- Create optimized database schema with vector columns
- Configure connection pooling and security settings
- Optimize PostgreSQL settings for SSD and 8GB RAM

### Phase 4: RAG Application Architecture (8 Sub-Phases)
**Objective:** Build object-oriented microservices following SOLID principles

#### Phase 4.1: FastAPI Structure & Configuration
**Files:** `app/main.py`, `app/config.py`

**Tasks:**
- Set up FastAPI application entry point with proper initialization
- Implement configuration management using environment variables  
- Configure CORS, middleware, and error handling
- Set up application lifecycle management (startup/shutdown)
- Implement basic health check endpoints
- Create logging configuration

#### Phase 4.2: Database Layer (PostgreSQL + pgvector)
**Files:** `app/core/database.py`

**Tasks:**
- Create async PostgreSQL connection pool management
- Design and implement database schema with vector columns
- Build CRUD operations for documents and metadata
- Implement vector index creation and management
- Add connection health monitoring and recovery
- Create database migration and initialization scripts
- Implement transaction management and rollback handling

#### Phase 4.3: LLaMA.cpp Client Integration
**Files:** `app/clients/llama_client.py`

**Tasks:**
- Build HTTP client for pi-worker llama.cpp communication
- Implement Phi-3.5-mini specific prompt formatting
- Create request/response handling with proper error management
- Add connection pooling and timeout configuration
- Implement health checks for LLM service availability
- Optimize request sizing for Q3_K_M memory constraints
- Add retry logic and circuit breaker patterns

#### Phase 4.4: Embedding Service Implementation
**Files:** `services/embedding_service.py`

**Tasks:**
- Create standalone FastAPI application for embedding service with its own main.py
- Implement configuration management with environment variables and service discovery
- Load and manage sentence-transformers model (all-MiniLM-L6-v2) with lazy loading
- Implement text-to-vector conversion with 384-dimensional output
- Create batch processing capabilities for multiple documents
- Design API endpoints for single text embedding and batch processing (/embed, /embed/batch)
- Create request/response models using Pydantic for API validation
- Add health check endpoints (/health, /info) for Kubernetes integration
- Add performance monitoring and metrics for embedding operations
- Implement error handling and logging specific to embedding operations

#### Phase 4.5: Vector Search & Retrieval
**Files:** `app/core/vector_search.py`

**Tasks:**
- Implement pgvector cosine similarity search queries
- Create configurable retrieval parameters (top-k, similarity threshold)
- Build result ranking and filtering algorithms
- Add metadata-based filtering capabilities
- Implement search result caching for performance
- Create search analytics and performance monitoring
- Add query optimization for different search patterns

#### Phase 4.6: Document Processing & Ingestion
**Files:** `app/core/document_processor.py`

**Tasks:**
- Implement multiple text chunking strategies (fixed-size, semantic, overlapping)
- Create file format parsers (PDF, DOCX, TXT, Markdown)
- Build metadata extraction and management system
- Implement batch document processing pipeline
- Add content cleaning and preprocessing capabilities
- Create progress tracking for large ingestion jobs
- Implement duplicate detection and handling

#### Phase 4.7: RAG Pipeline Orchestration
**Files:** `app/core/rag_pipeline.py`

**Tasks:**
- Design and implement the complete RAG workflow (Retrieval→Augmentation→Generation)
- Create service coordination logic between all components
- Build context construction from retrieved documents
- Implement prompt engineering for Phi-3.5-mini optimization
- Add response post-processing and formatting
- Create comprehensive error handling and fallback strategies
- Implement performance monitoring and pipeline analytics

#### Phase 4.8: API Endpoints & Error Handling
**Files:** `app/api/chat.py`, `app/api/documents.py`, `app/api/health.py`, `app/api/errors.py`

**Tasks:**
- Create chat endpoint with proper request/response models
- Implement document management endpoints (upload, list, delete)
- Build comprehensive health check and status endpoints
- Add request validation and sanitization
- Implement proper HTTP status codes and error responses
- Create API documentation with FastAPI automatic docs
- Add rate limiting and basic security measures
- Implement request/response logging and monitoring

### Phase 5: Implementation & Testing
**Objective:** Containerize and deploy complete system to k3s
- Create optimized Dockerfiles for ARM64 architecture
- Build multi-architecture Docker images
- Design Kubernetes manifests with resource limits and node selectors
- Configure persistent volumes and ConfigMaps
- Deploy full microservices stack to k3s cluster
- Implement health checks and readiness probes
- Conduct end-to-end testing of complete RAG pipeline
- Performance optimization and load testing

### Phase 6: Monitoring & Optimization
**Objective:** Add production-ready observability and optimization
- Implement metrics and monitoring endpoints in applications
- Deploy external monitoring stack (Prometheus/Grafana on development machine)
- Configure resource monitoring for both Raspberry Pis
- Set up log aggregation and alerting systems
- Create CI/CD pipeline with GitHub Actions or jenkins 
- Performance tuning based on actual usage patterns
- Documentation and maintenance procedures

### Phase 7: New PC Integration (Future)
**Objective:** Scale system with additional compute power
- Add 32GB PC (10.0.0.20) as additional k3s agent
- Deploy dual llama-server instances for concurrent users
- Implement load balancing between model servers
- Configure round-robin request distribution
- Performance testing with multiple concurrent users

### Phase 8: Advanced Adapters (Optional)
**Objective:** Implement LoRA fine-tuning for domain-specific improvements
- Set up adapter training environment and dependencies
- Prepare training datasets from existing knowledge base
- Train LoRA adapters on domain-specific data (NHS services, train schedules)
- Integrate adapter loading into llama.cpp deployment
- Modify FastAPI to support adapter-based inference modes
- Performance comparison: RAG vs Adapters vs Hybrid approaches
- Implement adapter swapping for different domains
- Optimize adapter training for Pi hardware constraints

---

## 🎯 Learning Outcomes

### Technical Competencies Gained:
1. **Distributed Systems** - Microservices communication and coordination
2. **Vector Databases** - Semantic search implementation and optimization
3. **Edge AI** - Local LLM deployment and optimization
4. **Kubernetes** - Container orchestration and resource management
5. **Async Programming** - High-performance Python applications
6. **Database Design** - Vector storage and similarity search
7. **API Development** - RESTful services with FastAPI
8. **DevOps Practices** - CI/CD, monitoring, and deployment automation

### Professional Skills Developed:
1. **System Architecture** - Designing scalable, maintainable systems
2. **Performance Optimization** - Resource-constrained optimization
3. **Documentation** - Technical writing and project communication
4. **Problem Solving** - Hardware constraint solutions
5. **Project Management** - Phased development approach

---


##  Core Success Metrics - Pi-Cluster RAG System

### Essential Performance Metrics

#### **1. Response Time**
- **Complete RAG Pipeline:** 2-3 minutes (Pi-only) / 30-60 seconds (with mini-PC)
- **Document Retrieval Only:** <1s (embedding + vector search)

#### **2. Memory Efficiency** 
- **Total System RAM:** 9-12GB (Pi-only, 3B model) / 12-15GB (with mini-PC, 8B model)
- **Stay within hardware limits** while maintaining functionality

#### **3. Concurrent Users**
- **Pi-only:** 1 simultaneous users with graceful degradation
- **With mini-PC:** 3-4 simultaneous users

#### **4. Document Retrieval Accuracy**
- **Precision:** >80% relevant documents in top-3 results
- **Query Success Rate:** >95% (excluding malformed requests)

#### **5. System Reliability**
- **Uptime:** 99%+ availability with proper error handling
- **Service Recovery:** <2 minutes for automatic restarts

#### **6. Data Scale**
- **Document Corpus:** 100-1,000 documents
- **Vector Database:** 1,000-10,000 embedding vectors

#### **7. Development Quality**
- **Deployment Success:** Smooth k3s deployments on ARM64
- **Error Handling:** Meaningful error messages and fallbacks