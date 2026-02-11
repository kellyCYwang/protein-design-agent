# Protein Design Agent — Project Summary

## Overview

The **Protein Design Agent** is an AI-powered research assistant for protein engineering and enzyme design. It combines **Retrieval-Augmented Generation (RAG)** over scientific literature with **agentic workflows** (LangGraph) and **external API integrations** to help researchers query databases, synthesize information from papers, and reason about complex biological questions.

---

## Key Functionalities

### 1. Intelligent Query Routing

The agent classifies incoming queries into three categories:

| Type | Description | Handler |
|------|-------------|---------|
| **Simple** | Direct lookups (EC number, structure) | Fast tool-only path |
| **Detailed** | Enzyme analysis, database searches | Tools + optional skill injection |
| **Research** | Broad literature questions, ML architectures | RAG retrieval → LLM synthesis |

A lightweight router LLM (e.g., `gpt-4o-mini`) handles classification with near-zero latency overhead.

### 2. RAG over Scientific Literature

| Feature | Description |
|---------|-------------|
| **Hybrid Search** | Vector similarity (PubMedBERT embeddings) + BM25 keyword search |
| **Reciprocal Rank Fusion** | Combines results from both methods (k=60) |
| **Multi-Backend** | ChromaDB (local) or Pinecone (cloud) |
| **PDF Processing** | PyMuPDF4LLM for layout-aware Markdown extraction |
| **Hierarchical Chunking** | Splits by headers, then paragraphs (~1000 chars) |
| **Image-Chunk Linking** | Extracts figures, links them to text chunks via regex detection |

### 3. MCP Server Integrations (Model Context Protocol)

| Server | Capabilities |
|--------|--------------|
| **EC** | Enzyme Commission number lookup by name |
| **PDB** | RCSB structure search (resolution, organism, method) |
| **UniProt** | Protein sequence, function, organism, GO annotations |
| **arXiv** | Search scientific papers (ML, biology, protein design) |
| **bioRxiv** | Preprint search + RCSB GraphQL for structure-related papers |

### 4. Skill System (Progressive Disclosure)

Skills are `.md` files in `agent/skills/` that define expert workflows:

- **Auto-discovered**: Files are registered at startup (`EnzymeAnalysis.md` → `enzyme_analysis`)
- **Router-selected**: Only injected when the query matches the skill
- **Extensible**: Add new skills without code changes

Example skill: **Enzyme Analysis** — gathers EC number, PDB structure, UniProt details, searches arXiv/bioRxiv for catalytic residues, and formats a structured report.

### 5. Conversation Memory (LangGraph Checkpointer)

- Uses `InMemorySaver` to persist conversation history per `thread_id`
- Enables multi-turn interactions ("What is AlphaFold3?" → "Give me the exact metrics")
- Thread IDs managed per Streamlit session

### 6. Streamlit Web Interface

- Chat-based UI with message history display
- Protein 3D structure visualization (stmol/py3Dmol)
- Session-based state management
- Reload button to reset conversation

### 7. Kubernetes-Ready Deployment

- Multi-stage Dockerfiles for app and embedding service
- Helm/Kustomize-compatible manifests
- PVCs for shared storage (papers, indexes, images)
- Ingress with WebSocket support for Streamlit
- Indexing Job/CronJob for PDF processing

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Streamlit 1.53+ |
| **LLM Orchestration** | LangGraph 1.0+ |
| **LLM Provider** | OpenAI-compatible (Zenmux/GPT-4o/Gemini) |
| **Embeddings** | PubMedBERT (`neuml/pubmedbert-base-embeddings`) via custom microservice |
| **Vector Database** | ChromaDB (local) / Pinecone (cloud) |
| **Keyword Search** | BM25 (rank_bm25) |
| **PDF Processing** | PyMuPDF + pymupdf4llm |
| **API Framework** | FastAPI (embedding service) |
| **Containerization** | Docker (multi-stage, non-root) |
| **Orchestration** | Kubernetes (Deployments, PVCs, Ingress) |
| **Language** | Python 3.11+ |

### Key Libraries

```
langchain, langgraph, langchain-openai
chromadb, pinecone
sentence-transformers
pymupdf, pymupdf4llm
rank-bm25
fastapi, uvicorn
streamlit, stmol, py3Dmol
```

---

## Current Limitations & Flaws

### 1. Memory & State Management

| Issue | Impact | Severity |
|-------|--------|----------|
| `InMemorySaver` only | Memory lost on pod restart/scaling | 🟡 Medium |
| No cross-session persistence | Users lose history after browser refresh | 🟡 Medium |
| Large conversations not trimmed | Token limits may be exceeded for long sessions | 🟠 Low |

### 2. RAG Pipeline

| Issue | Impact | Severity |
|-------|--------|----------|
| Embedding service is a bottleneck | Single point of failure, high latency for large batches | 🔴 High |
| BM25 index in-memory at runtime | Memory pressure with large corpora | 🟡 Medium |
| No incremental indexing | Must reindex entire corpus to add new PDFs | 🟡 Medium |
| Image extraction heuristics | Figure captions sometimes missed | 🟠 Low |
| No table content extraction | Tables rendered as images, not structured data | 🟡 Medium |

### 3. Tool Reliability

| Issue | Impact | Severity |
|-------|--------|----------|
| External API rate limits | arXiv/UniProt/PDB may throttle requests | 🟡 Medium |
| No retry logic with backoff | Transient failures cause tool errors | 🟡 Medium |
| Cached lookups don't expire | Stale data for rarely-updated entries | 🟠 Low |

### 4. Skill System

| Issue | Impact | Severity |
|-------|--------|----------|
| Single skill per query | Cannot combine skills for complex queries | 🟠 Low |
| No skill versioning | Breaking changes affect all users | 🟠 Low |
| Description extraction is naive | First non-heading line may not be descriptive | 🟠 Low |

---

## Scaling Challenges

### Scenario: 100+ Concurrent Users

| Problem | Cause | Mitigation |
|---------|-------|------------|
| **Embedding latency** | Single embedding service, model loading time | Scale to multiple replicas, use GPU inference |
| **Memory exhaustion** | InMemorySaver, BM25 index per pod | Switch to Redis/PostgreSQL for checkpoints, centralize BM25 |
| **ChromaDB contention** | SQLite not designed for high concurrency | Migrate to Pinecone or hosted ChromaDB |
| **PVC I/O bottleneck** | ReadWriteMany NFS may have limited IOPS | Use faster storage (EBS gp3, local SSD) or object storage |

### Scenario: 10,000+ Papers

| Problem | Cause | Mitigation |
|---------|-------|------------|
| **Indexing time** | Linear with corpus size, no parallelism | Implement parallel chunk processing, distributed embeddings |
| **Search latency** | BM25 tokenization + vector search | Pre-compute BM25 scores, use approximate NN (HNSW tuning) |
| **Storage costs** | ChromaDB + images + BM25 index | Compress embeddings, use tiered storage |

### Scenario: Multi-Region Deployment

| Problem | Cause | Mitigation |
|---------|-------|------------|
| **Vector DB sync** | ChromaDB local to each region | Use Pinecone (global) or implement replication |
| **Conversation state** | InMemorySaver not distributed | Redis Cluster or managed state store |
| **Embedding model** | Downloaded on each pod startup | Pre-bake model into image or use shared PVC |

---

## Future Improvements

### High Priority

1. **Persistent Conversation Memory**
   - Replace `InMemorySaver` with `RedisSaver` or `PostgresSaver`
   - Enable cross-session history retrieval

2. **Streaming Responses**
   - Implement token-by-token streaming in Streamlit
   - Show tool calls in real-time

3. **Incremental Indexing**
   - Track indexed files by hash
   - Only process new/modified PDFs

4. **GPU Acceleration for Embeddings**
   - Serve embedding model on GPU (CUDA/ROCm)
   - Batch requests for throughput

### Medium Priority

5. **Table Extraction**
   - Use table detection models (TableTransformer)
   - Store structured table data for querying

6. **Multi-Skill Composition**
   - Allow router to select multiple skills
   - Chain skill outputs

7. **Tool Retry & Circuit Breaker**
   - Exponential backoff for external APIs
   - Graceful degradation when services are down

8. **Observability**
   - OpenTelemetry tracing
   - LangSmith integration for LLM debugging

### Low Priority

9. **Fine-Tuned Embeddings**
   - Train on protein-specific corpus
   - Improve retrieval for domain terminology

10. **Multi-Modal RAG**
    - Embed images alongside text (CLIP)
    - Enable "find papers with similar figures"

11. **User Authentication**
    - OAuth2/OIDC integration
    - Per-user index namespaces

12. **Plugin System**
    - Dynamic tool loading
    - Third-party skill marketplace

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USERS (~10-100)                                │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ HTTPS
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Kubernetes Ingress (NGINX)                          │
│                         protein-rag.your-domain.com                         │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     protein-rag-app (Streamlit + LangGraph)                 │
│   ┌───────────────┐    ┌───────────────┐    ┌───────────────────────────┐  │
│   │  Router LLM   │───▶│  Worker LLM   │───▶│     Tool Execution        │  │
│   │ (gpt-4o-mini) │    │   (gpt-4o)    │    │ (EC/PDB/UniProt/arXiv)    │  │
│   └───────────────┘    └───────────────┘    └───────────────────────────┘  │
│          │                                              │                   │
│          │ classify query                              │ external APIs     │
│          ▼                                              ▼                   │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │                    RAG Backend (search_research_papers)           │    │
│   │   ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐    │    │
│   │   │ ChromaDB or │ + │   BM25      │ = │   RRF Fusion        │    │    │
│   │   │  Pinecone   │   │  (local)    │   │   (top-k results)   │    │    │
│   │   └─────────────┘   └─────────────┘   └─────────────────────┘    │    │
│   └───────────────────────────────────────────────────────────────────┘    │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
            ▼                       ▼                       ▼
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│ embedding-service │   │   PVC: papers/    │   │ PVC: chroma_db/   │
│   (PubMedBERT)    │   │      (PDFs)       │   │      bm25/        │
│   FastAPI:8000    │   │                   │   │    pdf_images/    │
└───────────────────┘   └───────────────────┘   └───────────────────┘
```

---

## Conclusion

The Protein Design Agent is a **production-ready foundation** for AI-assisted protein research. Its modular architecture (MCP servers, skill system, swappable RAG backends) enables rapid iteration. The primary scaling bottlenecks are the embedding service and in-memory state management, both of which have clear upgrade paths.

For teams of ~10 researchers with a few hundred papers, the current architecture is sufficient. For larger deployments, prioritize GPU embeddings, Redis-based checkpointing, and Pinecone for vector storage.
