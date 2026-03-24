# Protein Design Research Agent

An AI-powered assistant for protein engineering and enzyme design research. Integrates **RAG (Retrieval-Augmented Generation)** with **agentic workflows** (LangGraph) and a modern React interface for querying protein databases, searching literature, and visualizing 3D structures.

---

## Features

### Intelligent Query Routing
The LangGraph agent classifies every query and routes it through the appropriate path:
- **Simple** — fast EC number lookups via a lightweight router model
- **Detailed** — full tool-calling workflow (PDB, UniProt, ExplorEnz, arXiv, biorXiv)
- **Research** — hybrid RAG over local PDF library, then detailed analysis

### Hybrid RAG Pipeline
- **Vector search** via PubMedBERT embeddings (`neuml/pubmedbert-base-embeddings`)
- **BM25 keyword search** fused with Reciprocal Rank Fusion (k=60)
- PDF processing with **Docling** → Markdown → `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter`
- Supports both **ChromaDB** (local) and **Pinecone** (cloud) backends
- Chunks are prefixed with `Paper Title | Section` for context enrichment

### MCP-style Tool Servers
Modular servers in `mcp_servers/` each expose search/lookup methods:

| Server | Data source |
|--------|-------------|
| `ec/` | ExplorEnz — EC number classification |
| `pdb/` | RCSB GraphQL — structure data |
| `uniprot/` | UniProt REST — sequences & annotations |
| `arxiv/` | arXiv Atom feed |
| `biorxiv/` | EuropePMC REST |

### Skill System
Drop a Markdown file into `agent/skills/` and the router automatically gains the ability to select it. The file's content is injected as a system prompt for the detailed handler — no code changes needed.

### React Frontend
A custom React/TypeScript/Vite UI replaces the previous Streamlit app, served through a FastAPI SSE streaming backend.

- Live **pipeline status bar** showing agent routing → tools → response in real time
- Full **Markdown rendering** with syntax-highlighted code blocks
- Automatic **PDB ID detection** in responses with one-click structure loading
- **3D protein viewer** (3Dmol.js) with rainbow cartoon rendering
  - Drag the left border to resize the panel (280–900 px)
  - **Sequence strip** showing residues color-coded by type (hydrophobic / polar / charged± / glycine)
  - Click any residue to highlight it in the 3D view with a gold sphere overlay
- Persistent conversation memory via `SqliteSaver` (per session `thread_id`)

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, TypeScript, Vite |
| Styling | Pure CSS (IBM Plex Mono + Cormorant Garamond) |
| 3D Viewer | 3Dmol.js (CDN) |
| API server | FastAPI + Uvicorn/Gunicorn (SSE streaming) |
| Agent | LangGraph `StateGraph`, `SqliteSaver` checkpointer |
| LLM | OpenAI-compatible (Zenmux) — configurable router + worker models |
| RAG — local | ChromaDB + BM25 |
| RAG — cloud | Pinecone + BM25 |
| Embeddings | PubMedBERT microservice (FastAPI, port 8000) |
| PDF processing | Docling → Markdown |
| Observability | LangSmith (optional) |

---

## Project Structure

```
protein-design-agent/
├── agent/
│   ├── agent.py              # LangGraph StateGraph + query routing
│   ├── skills/               # Markdown skill files (auto-discovered)
│   └── rag/
│       ├── chroma_rag.py
│       ├── pinecone_rag.py
│       ├── pdf_processing.py
│       └── rag_cli.py
├── mcp_servers/
│   ├── ec/server.py
│   ├── pdb/server.py
│   ├── uniprot/server.py
│   ├── arxiv/server.py
│   └── biorxiv/server.py
├── frontend/                 # React/TypeScript/Vite app
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── Sidebar.tsx
│   │   │   ├── ChatArea.tsx
│   │   │   ├── MessageBubble.tsx
│   │   │   ├── PipelineStatus.tsx
│   │   │   └── ProteinViewer.tsx
│   │   ├── hooks/useChat.ts  # SSE streaming + agent state
│   │   ├── types/index.ts
│   │   └── styles/globals.css
│   ├── package.json
│   └── vite.config.ts        # proxies /api → localhost:8001
├── deploy/                   # All deployment files
│   ├── Dockerfile            # Python API image
│   ├── Dockerfile.embedding  # PubMedBERT embedding service
│   ├── Dockerfile.frontend   # Node build → nginx
│   ├── docker-compose.prod.yml
│   └── nginx/nginx.conf      # SSE-aware reverse proxy config
├── api.py                    # FastAPI SSE backend (port 8001)
├── app.py                    # Legacy Streamlit app
├── embedding_service.py      # PubMedBERT microservice (port 8000)
├── data/
│   ├── papers/               # Place PDFs here
│   ├── chroma_db/
│   ├── bm25/
│   └── checkpoints.db        # SqliteSaver conversation history
└── pyproject.toml
```

---

## Setup

### Prerequisites
- Python 3.13+
- Node.js 20+
- `uv` (recommended) or `pip`

### 1. Install Python dependencies

```bash
uv sync
```

### 2. Configure environment

Create `.env` in the project root:

```env
ZENMUX_API_KEY=...
ZENMUX_BASE_URL=...          # defaults to https://api.zenmux.com/v1
ROUTER_MODEL=gpt-4o-mini
WORKER_MODEL=gpt-4o
RAG_BACKEND=chroma           # or pinecone
EMBEDDING_SERVICE_URL=http://localhost:8000/embed
PINECONE_API_KEY=...         # only needed for Pinecone backend
PINECONE_INDEX_NAME=paper-rag-index
LANGCHAIN_API_KEY=...        # optional, enables LangSmith tracing
```

### 3. Install frontend dependencies

```bash
cd frontend && npm install
```

---

## Running Locally

Three processes are needed. Open three terminals from the project root:

**Terminal 1 — Embedding microservice** (required for RAG):
```bash
python embedding_service.py
# Serves PubMedBERT on http://localhost:8000
```

**Terminal 2 — FastAPI backend**:
```bash
uvicorn api:app --reload --port 8001
```

**Terminal 3 — React dev server**:
```bash
cd frontend && npm run dev
# Opens http://localhost:5173
# /api/* is proxied to the FastAPI backend automatically
```

### Index PDFs (one-time, after adding papers to `data/papers/`)

```bash
# ChromaDB (default)
python -m agent.rag.rag_cli --backend chroma --index

# Pinecone
python -m agent.rag.rag_cli --backend pinecone --index

# Force full reindex
python -m agent.rag.rag_cli --backend chroma --index --reindex

# Verify
python -m agent.rag.rag_cli --backend chroma --search "RFDiffusion architecture"
```

---

## Deployment (Docker, ~100 users)

All deployment files live in `deploy/`. The recommended target is a **4 vCPU / 8 GB RAM** VM.

```bash
# From the project root
docker compose -f deploy/docker-compose.prod.yml up -d --build

# Index PDFs after first deploy
docker compose -f deploy/docker-compose.prod.yml exec api \
  python -m agent.rag.rag_cli --backend chroma --index
```

The compose stack:
- **nginx** — serves the built React app as static files and reverse-proxies `/api/` to the FastAPI service (SSE buffering disabled)
- **api** — 4 Gunicorn/Uvicorn workers, SQLite WAL for shared conversation state
- **embedding-service** — PubMedBERT model, cached to a named volume

See `deploy/nginx/nginx.conf` for SSL configuration notes.

---

## Example Queries

| Type | Query |
|------|-------|
| Simple | "What's the EC number for chalcone isomerase?" |
| Detailed | "Explain the catalytic mechanism of chymotrypsin" |
| Structure | "Tell me about lysozyme" → opens 3D viewer automatically |
| Research | "What is the model architecture of RFDiffusion?" |
| UniProt | "Get the amino acid sequence of human hexokinase-1" |
