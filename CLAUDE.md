# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install dependencies** (Python 3.13 required):
```bash
uv sync
```

**Run the FastAPI backend:**
```bash
uvicorn api:app --reload --port 8001
```

**Run the React/TypeScript frontend:**
```bash
cd frontend && npm run dev
# Runs on port 5173, proxies /api to localhost:8001
```

**Start the embedding microservice** (required before indexing or running RAG queries):
```bash
python embedding_service.py
# Runs on port 8000 by default; override with EMBEDDING_PORT env var
# RAG expects it at http://localhost:8011/embed — set EMBEDDING_SERVICE_URL accordingly
```

**Index PDFs** (place files in `data/papers/` first):
```bash
# Chroma backend (default)
python -m agent.rag.rag_cli --backend chroma --index

# Pinecone backend
python -m agent.rag.rag_cli --backend pinecone --index

# Force full reindex
python -m agent.rag.rag_cli --backend chroma --index --reindex

# Search / stats
python -m agent.rag.rag_cli --backend chroma --search "RFDiffusion architecture"
python -m agent.rag.rag_cli --stats
```

**Run the agent from the command line** (bypasses Streamlit):
```bash
python agent/agent.py
```

**Build Docker images:**
```bash
docker build -t protein-design-agent .
docker build -f Dockerfile.embedding -t protein-embedding-service .
```

## Environment Variables

Create a `.env` file in the project root:

```
ZENMUX_API_KEY=...          # Required — OpenAI-compatible API key
ZENMUX_BASE_URL=...         # Optional, defaults to https://api.zenmux.com/v1
ROUTER_MODEL=gpt-4o-mini    # Optional, model used for query classification
WORKER_MODEL=gpt-4o         # Optional, model used for detailed responses/tool calls
RAG_BACKEND=chroma          # 'chroma' (default) or 'pinecone'
EMBEDDING_SERVICE_URL=http://localhost:8011/embed
PINECONE_API_KEY=...        # Required for Pinecone backend
PINECONE_INDEX_NAME=paper-rag-index
LANGCHAIN_API_KEY=...       # Optional, for LangSmith tracing
```

## Architecture

### Agent Graph (`agent/agent.py`)

The core is a **LangGraph `StateGraph`** with `AgentState` holding `messages`, `query_type`, `skill_name`, `needs_rag`, and `rag_context`.

Query flow:
1. **`route_query`** — router LLM classifies query as `simple` | `detailed` | `research`, and selects a skill from `SKILL_REGISTRY` if applicable.
2. **`simple_handler`** — calls `get_ec_number` via the smaller router LLM; loops through `tools` node until done.
3. **`rag_handler`** — only for `research` queries; runs `search_research_papers` and stores results in `rag_context`, then continues to `detailed_handler`.
4. **`detailed_handler`** — worker LLM with all tools; injects a skill's system prompt if one was selected, otherwise uses a generic protein engineering prompt.
5. **`tools`** (LangGraph `ToolNode`) — executes any tool calls; routes back to `simple_handler` or `detailed_handler`.

Conversation memory uses `SqliteSaver` (persisted to `./data/checkpoints.db`, per `thread_id`). Each Streamlit session gets a UUID `thread_id`.

### RAG Pipeline

Two swappable backends share the same `agent/rag/pdf_processing.py` utilities:

- **`ChromaPDFRAG`** (`agent/rag/chroma_rag.py`) — local ChromaDB + BM25; default.
- **`PineconePDFRAG`** (`agent/rag/pinecone_rag.py`) — Pinecone cloud vector store + BM25.

Hybrid search combines **vector similarity** (via the PubMedBERT embedding microservice) and **BM25 keyword search**, fused with **Reciprocal Rank Fusion (k=60)**. BM25 index is serialized to `data/bm25/bm25.pkl`.

PDF processing (`agent/rag/pdf_processing.py`): PyMuPDF4LLM → Markdown → `MarkdownHeaderTextSplitter` → `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap). Each chunk is prefixed with `Paper Title: ... | Section: ...` for context enrichment.

The embedding microservice (`embedding_service.py`) is a **separate FastAPI process** serving `neuml/pubmedbert-base-embeddings` on `/embed`. Both backends call it over HTTP — it must be running before indexing or querying.

### MCP Servers (`mcp_servers/`)

Each subdirectory (`ec/`, `pdb/`, `uniprot/`, `arxiv/`, `biorxiv/`) contains a `server.py` with a class exposing search/lookup methods. These are instantiated as singletons in `agent/agent.py` and wrapped with `@tool` decorators. They call external APIs (ExplorEnz, RCSB GraphQL, UniProt REST, arXiv Atom, EuropePMC).

### Skill System (`agent/skills/`)

Skills are **Markdown files** in `agent/skills/`. At import time, `_build_skill_registry()` auto-discovers all `.md` files and derives snake_case keys (e.g., `EnzymeAnalysis.md` → `enzyme_analysis`). When the router selects a skill, its full Markdown content is injected as a `SystemMessage` into `detailed_handler`. Add new skills by dropping `.md` files in that directory — no code changes needed.

### FastAPI Backend (`api.py`)

- Streaming SSE endpoint at `POST /api/chat` that runs the LangGraph agent in a background thread and yields `route`, `status`, `tool`, `response`, `done`, and `error` events.
- `POST /api/chat/cancel` signals cancellation for an active stream by `thread_id`.
- Query-level caching (exact + semantic similarity) via Redis.

### React/TypeScript Frontend (`frontend/`)

- Vite + React 18 + TypeScript.
- `useChat` hook manages SSE streaming, cancel via `AbortController` + backend cancel endpoint, and thread persistence.
- `ChatArea` component with send/cancel button toggle and `Escape` key shortcut.
- `ProteinViewer` renders interactive 3D structures via 3Dmol.js, auto-detected from `PDB ID: XXXX` patterns in responses.
- `PipelineStatus` shows agent progress through Route → Research → Tool → Respond stages.

### Kubernetes (`k8s/`)

Kustomize-based manifests deploy two `Deployment`s (app + embedding-service) with shared PVCs for papers, indexes, and images. A `Job`/`CronJob` handles PDF indexing.

## Known Issues

- BM25 does not support incremental updates in the Pinecone backend when called via `index_local_pdfs`; each call rebuilds from scratch (acceptable for batch use).
- `SqliteSaver` conversation history is keyed by `thread_id`, which currently resets on browser reload — the data is durable on disk but re-opening a tab starts a fresh thread.
