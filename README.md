# Pi-Cluster RAG Chatbot
**A hands-on project building distributed AI systems with Kubernetes and Raspberry Pi**

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Kubernetes](https://img.shields.io/badge/kubernetes-%23326ce5.svg?style=for-the-badge&logo=kubernetes&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Raspberry Pi](https://img.shields.io/badge/-RaspberryPi-C51A4A?style=for-the-badge&logo=Raspberry-Pi)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![PostgreSQL](https://img.shields.io/badge/postgresql-%23316192.svg?style=for-the-badge&logo=postgresql&logoColor=white)

## Project Objectives
- **Master distributed AI systems** by building from scratch with commodity hardware
- **Develop Kubernetes expertise** through hands-on cluster management and orchestration
- **Explore performance optimization** in resource-constrained environments
- **Practice DevOps workflows** with containerization, deployment, and monitoring

## Project Documentation

This project demonstrates systematic software engineering through structured planning and execution:

**Planning & Architecture:**
- [Complete Project Plan](docs/project_plan.md) - 6-phase development architecture with detailed technical specifications

### Key Achievements
- **20x performance improvement** through quantization research and optimization
- **Custom ARM compilation** solving real-world compatibility challenges  
- **Production-ready Kubernetes deployment** across diverse hardware
- **Scalable architecture design** ready for future expansion

---

## Microservice Implementation

### Embedding Service API 

A complete, microservice built with FastAPI.

#### **Key Features**
- **Live API Endpoints**: `/embed`, `/embed/batch`, `/health`, `/info`
- **ML Integration**: Sentence-transformers model serving with 384-dimensional vectors
- **Production Patterns**: Dependency injection, structured logging, comprehensive error handling
- **Kubernetes Integration**: Health checks designed for readiness/liveness probes
- **Resource Monitoring**: Memory usage tracking and uptime metrics
- **Operational Design**: Graceful startup/shutdown with model preloading

#### **Source Code & Deployment**
- **API Implementation**: [app/services/embedding/main.py](app/services/embedding/main.py)
- **Core Service Logic**: [app/services/embedding/core/](app/services/embedding/core/)
- **Data Models**: [app/services/embedding/core/models.py](app/services/embedding/core/models.py)
- **Docker Configuration**: [docker/services/embedding/embedder.amd64.Dockerfile](docker/services/embedding/embedder.amd64.Dockerfile)

#### **API Capabilities**
```bash
# Single text embedding
curl -X POST "http://pi-cluster:8001/embed" \
  -H "Content-Type: application/json" \
  -d '{"text": "Machine learning on edge devices"}'

# Batch processing
curl -X POST "http://pi-cluster:8001/embed/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Text one", "Text two", "Text three"]}'

# Health monitoring
curl "http://pi-cluster:8001/health"
```

#### **Software Engineering Highlights**
- **Pydantic Models**: Type-safe request/response validation with automatic OpenAPI documentation
- **Error Handling**: Proper HTTP status codes (400 for validation, 500 for server errors)
- **Async Architecture**: Non-blocking request handling for scalability
- **Configuration Management**: Environment-based settings with sensible defaults
- **Logging Strategy**: Structured logging with context for operational visibility

#### **Production Readiness**
- **Containerized Deployment**: Running in k3s with resource constraints
- **Health Monitoring**: Kubernetes-compatible endpoints for cluster management
- **Memory Efficiency**: Optimized for 4GB Pi hardware with usage tracking
- **Batch Optimization**: Efficient processing for multiple embedding requests
- **Graceful Failures**: Comprehensive exception handling with detailed error responses

**This microservice demonstrates the complete software development lifecycle**: from design and implementation to containerization and k3s deployment in a distributed system.

---

## Technical Achievements 

- **Production Kubernetes Cluster**: Successfully deployed k3s across diverse Pi hardware with service discovery and pod orchestration
- **Custom ARM64 Docker Images**: Built optimized containers with hand-compiled llama.cpp for ARM architecture 
- **AI Model Deployment**: Deployed 3B parameter Phi-3.5-mini model with inference serving at 4 tokens/sec
- **Performance Optimization**: Achieved 20x improvement through Q3_K_M quantization research and implementation
- **Production Microservice**: Built and deployed embedding API with sentence-transformers integration, featuring comprehensive error handling, health monitoring, and Kubernetes-ready design
- **End-to-End Software Delivery**: Complete working system from ML model serving to REST API with proper logging and operational monitoring

## Development Approach & Modern Tooling

Built using AI-assisted development practices alongside traditional software engineering. AI tools provided:

**Research & Implementation Acceleration**
- ARM compilation techniques and quantization method analysis
- Initial FastAPI service implementations and Docker configurations  
- Kubernetes networking patterns and distributed systems concepts

**Engineering Focus Areas**
- Distributed architecture design separating compute and I/O operations
- Technology selection: k3s vs full Kubernetes, Q3_K_M quantization strategy
- Performance optimization methodology achieving 20x improvement
- Production-ready microservices with comprehensive error handling

All AI-generated components required extensive testing, debugging, and integration work. This approach enabled focus on high-level system design while accelerating development.

## Future Goals

- **Vector Database Integration**: Complete PostgreSQL + pgvector setup for semantic search capabilities
- **End-to-End RAG Pipeline**: Integrate document processing, embedding generation, and retrieval-augmented responses
- **Production Deployment**: Implement monitoring with Prometheus/Grafana, CI/CD pipelines, and containerized database services
- **Advanced Features**: Explore LoRA adapter training for domain-specific model optimization

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Pi-Cluster RAG System                               │
│                        Implementation Roadmap                               │
└─────────────────────────────────────────────────────────────────────────────┘

Phase 1: Foundation - COMPLETE
┌─────────────────────────────────┐     ┌─────────────────────────────────┐
│         pi-master               │───▶│         pi-worker               │
│      (Pi 5, 8GB RAM)            │     │      (Pi 5, 4GB RAM)            │
├─────────────────────────────────┤     ├─────────────────────────────────┤
│ [X] k3s Server                  │     │ [X] k3s Agent                   │
│     • Cluster management        │     │     • Container runtime         │
│                                 │     │     • Pod scheduling            │
│                                 │     │                                 │
│                                 │     │ [X] llama.cpp                   │
│                                 │     │     • Phi-3.5-mini (3B)         │
│                                 │     │     • Q3_K_M quantization       │
│                                 │     │     • 4 tokens/sec inference    │
│                                 │     │     • ARM64 optimized           │
└─────────────────────────────────┘     └─────────────────────────────────┘

Phase 2: RAG Pipeline - IN PROGRESS
┌─────────────────────────────────┐     ┌─────────────────────────────────┐
│         pi-master               │───▶│         pi-worker               │
│      (Pi 5, 8GB RAM)            │     │      (Pi 5, 4GB RAM)            │
├─────────────────────────────────┤     ├─────────────────────────────────┤
│ [>] FastAPI Gateway             │     │    [X] k3s Agent                │
│     • RAG orchestration         │     │       • Container runtime       │
│     • Request routing           │     │       • Pod scheduling          │
│     • Response aggregation      │     │                                 │
│                                 │     │   [X] llama.cpp                 │
│ [ ] PostgreSQL + pgvector       │     │     • Inference engine          │
│     • Vector embeddings         │     │     • Model serving             │
│     • Semantic search           │     │                                 │
│     • Document metadata         │     │                                 │
│                                 │     │                                 │
│ [ ] Embedding Service           │     │                                 │
│     • Document processing       │     │                                 │
│     • Vector generation         │     │                                 │
└─────────────────────────────────┘     └─────────────────────────────────┘
  
Legend:
[X] Complete    [>] In Progress  [ ] Planned


RAG Pipeline Flow (Target):
┌───────────────────────────────────────────────────────────────────────────────┐
│  Question  →  Embedding  →  Vector Search  →  Context  →  LLM Generation      │
│                                                                               │
│  User Input   →   Encode Query  →   Find Similar   →   Retrieve   →  Generate │
│               →   to Vector    →   Documents      →   Context    →  Response  │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## Technical Challenges

<details>
<summary><strong>ARM Architecture Compilation</strong></summary>

**Challenge**: Standard Docker images don't work on ARM-based Raspberry Pi hardware.

**Research & Solution**: 
- Studied ARM64 architecture differences and compilation requirements
- Built custom ARM64 Docker images with hand-picked dependencies
- Compiled llama.cpp from source with ARM optimizations
- Minimized image size through multi-stage builds and dependency analysis

**Skills Developed**: Cross-platform compilation, Docker optimization, ARM architecture understanding.
</details>

<details>
<summary><strong>Memory & Performance Optimization</strong></summary>

**Challenge**: 4GB RAM constraint causing extremely slow inference (0.2 tokens/sec).

**Research & Solution**: 
- Deep-dived into quantization techniques and their trade-offs
- Experimented with Q4 vs Q3 quantization methods
- Analyzed I/O bottlenecks between shared SSD and worker node
- Redesigned hardware allocation strategy based on performance findings

**Skills Developed**: Model optimization, performance profiling, hardware resource management.
</details>

<details>
<summary><strong>Distributed Architecture Design</strong></summary>

**Challenge**: Efficiently distribute compute-intensive AI workloads across diverse hardware.

**Research & Solution**: 
- Studied Kubernetes networking and service discovery patterns
- Implemented pod-to-pod communication across different node types
- Separated CPU-intensive inference from I/O-intensive operations

**Skills Developed**: Distributed systems design, Kubernetes networking, microservices architecture.
</details>

---

## Technology Stack & Focus areas

### **Core Infrastructure (Primary Area)**
- **Orchestration**: k3s (Lightweight Kubernetes) - cluster management, networking
- **Containerization**: Docker with custom ARM64 images - cross-platform development
- **Hardware**: Raspberry Pi 5 cluster

### **AI/ML Stack (Secondary Area)**
- **Model Serving**: llama.cpp with Phi-3.5-mini (3B parameters) - inference optimization
- **Quantization**: Q3_K_M for memory efficiency - performance tuning
- **Embeddings**: sentence-transformers - vector operations
- **Vector Search**: PostgreSQL + pgvector - database design

### **Application Layer (Development Skills)**
- **API Framework**: Python FastAPI with async operations - modern web APIs
- **Database**: PostgreSQL with vector extensions - data persistence
- **Document Processing**: pypdf, python-docx, beautifulsoup4 - data ingestion

---

## Metrics & Achievements

| Metric | Achievement |
|--------|---------------------|
| **Performance Optimization** | 20x improvement (0.2→4 tokens/sec) |
| **Memory Management** | Fit 3B model in 4GB RAM |
| **Cluster Management** | 2-node diverse cluster |
| **Container Optimization** | Custom ARM64 builds |
| **Architecture Evolution** | Monolith → microservices |

---

## Skills Developed Through This Project

### **Distributed Systems & Infrastructure**
- Hands-on Kubernetes cluster management and service orchestration
- Container orchestration across diverse hardware environments
- Network configuration, service discovery, and inter-pod communication
- Resource allocation and scheduling in constrained environments

### **AI/ML Engineering & Optimization**
- Model quantization research and practical implementation
- ARM-based AI inference deployment and optimization
- RAG (Retrieval-Augmented Generation) pipeline architecture
- Performance profiling and optimization in resource-constrained environments

### **DevOps & Platform Engineering**
- Custom Docker image creation and multi-stage optimization
- Cross-platform compilation and dependency management
- Infrastructure as Code with Kubernetes manifests
- Monitoring and performance analysis workflows

---

## Next Objectives

### **Current Goals**
- **API Development**: Complete FastAPI integration with async patterns
- **Vector Operations**: Implement semantic search with pgvector
- **System Integration**: Build end-to-end RAG pipeline


### **Advanced Topics (Future objectives)**
- **Observability**: Monitoring and alerting with Prometheus/Grafana dashboards
- **Automation**: CI/CD pipeline development with GitHub Actions
- **Model Adaptation**: LoRA adapter training for domain-specific optimization

---

## Environment Setup

### **Hardware Platform**
- 2x Raspberry Pi 5 (4GB + 8GB RAM variants) - resource constraint challenges
- Shared SSD storage with multiple partitions - storage architecture
- Network connectivity for cluster communication - distributed systems

### **Software Stack**
- Ubuntu 24.04.2 Server (ARM64) - Linux system administration
- k3s (Kubernetes distribution) - container orchestration
- Docker with ARM64 support - containerization and optimization
- Python 3.9+ with FastAPI ecosystem - modern web development
