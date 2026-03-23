# RAG Module

Retrieval-Augmented Generation backends for the Protein Design Agent. This module provides hybrid vector + BM25 search over locally indexed PDF papers, with support for both ChromaDB (local) and Pinecone (managed cloud) backends.

## Features

- **Hybrid Search**: Combines vector similarity with BM25 keyword search using Reciprocal Rank Fusion
- **Image Extraction**: Automatically extracts images from PDFs and saves them to disk
- **Figure-Chunk Linking**: Detects figure/table references in text and links chunks to their associated images
- **Multi-Backend Support**: ChromaDB (local) and Pinecone (cloud) with identical interfaces

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     PDF Processing                           │
│  pdf_processing.py                                           │
│  - extract_metadata_from_pdf()                               │
│  - extract_text_as_markdown()  (PyMuPDF4LLM)                │
│  - extract_images_from_pdf()   (saves images to disk)       │
│  - chunk_text()  (Header splitting + figure/table detection)│
│  - get_embeddings()  (via embedding microservice)           │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
┌─────────────────────┐     ┌─────────────────────────┐
│   ChromaPDFRAG      │     │    PineconePDFRAG       │
│   chroma_rag.py     │     │    pinecone_rag.py      │
│                     │     │                         │
│   - ChromaDB store  │     │   - Pinecone cloud      │
│   - Local BM25      │     │   - Optional local BM25 │
│   - Hybrid RRF      │     │   - Hybrid RRF          │
│   - Image linking   │     │   - Image linking       │
└─────────────────────┘     └─────────────────────────┘
```

## Quick Start

### Using the CLI

```bash
# Index PDFs with ChromaDB (default)
python -m agent.rag.rag_cli --backend chroma --index

# Index PDFs with Pinecone
python -m agent.rag.rag_cli --backend pinecone --index

# Search
python -m agent.rag.rag_cli --backend chroma --search "RFDiffusion architecture"

# Get statistics
python -m agent.rag.rag_cli --stats
```

### Using in Python

```python
from agent.rag import ChromaPDFRAG, PineconePDFRAG

# ChromaDB backend (local)
rag = ChromaPDFRAG(
    pdf_directory="./data/papers",
    vd_persist_directory="./data/chroma_db",
    bm25_persist_directory="./data/bm25",
)

# Or Pinecone backend (cloud)
rag = PineconePDFRAG(
    pdf_directory="./data/papers",
    pinecone_api_key="your-api-key",
    pinecone_index_name="paper-rag-index",
)

# Index and search
rag.index_local_pdfs()
results = rag.search("protein design methods", top_k=5)
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RAG_BACKEND` | Backend to use: `chroma` or `pinecone` | `chroma` |
| `RAG_PDF_DIR` | Directory containing PDF files | `./data/papers` |
| `RAG_CHROMA_DIR` | ChromaDB persist directory | `./data/chroma_db` |
| `RAG_BM25_DIR` | BM25 index persist directory | `./data/bm25` |
| `EMBEDDING_SERVICE_URL` | Embedding microservice URL | `http://localhost:8011/embed` |
| `PINECONE_API_KEY` | Pinecone API key (required for pinecone) | - |
| `PINECONE_INDEX_NAME` | Pinecone index name | `paper-rag-index` |
| `PINECONE_NAMESPACE` | Pinecone namespace | `papers` |
| `PINECONE_CLOUD` | Pinecone cloud provider | `aws` |
| `PINECONE_REGION` | Pinecone region | `us-east-1` |
| `RAG_HYBRID_SEARCH` | Enable hybrid search (vector + BM25) | `true` |

## How Search Works

### ChromaDB Backend

1. **Vector Search**: Query is embedded and searched against ChromaDB using cosine similarity
2. **BM25 Search**: Query is tokenized and searched against a local BM25 index
3. **Reciprocal Rank Fusion (RRF)**: Results from both searches are combined using RRF with k=60
4. **Result Mapping**: Final top-k results are fetched with full content and metadata

### Pinecone Backend

1. **Vector Search**: Query is embedded and searched against Pinecone index
2. **BM25 Search** (optional): If hybrid search is enabled, query is also searched against a local BM25 index built during indexing
3. **Reciprocal Rank Fusion**: Results are combined using RRF (same as ChromaDB)
4. **Result Mapping**: Metadata is retrieved from Pinecone matches or fetched for BM25-only results

## Image-Chunk Linking

During indexing, the system:

1. **Extracts images** from each PDF page and saves them to `./data/pdf_images/<paper_id>/`
2. **Detects figure captions** near each image (e.g., "Figure 3", "Fig. 2A")
3. **Detects figure/table references** in chunk text using regex patterns
4. **Links chunks to images** by matching figure IDs

### Search Results Include:

Each search result now contains:

```python
{
    "title": "Paper Title",
    "section": "Methods > Protein Design",
    "content": "...",
    "score": 0.85,
    "has_table": True,
    "has_figure": True,
    "figure_ids": ["Figure 1", "Figure 2A"],  # References in this chunk
    "table_ids": ["Table 1"],                  # Table references
    "image_paths": [                           # Linked image files
        "data/pdf_images/paper/page_3_img_0.png",
        "data/pdf_images/paper/page_3_img_1.png"
    ]
}
```

### Usage Example

```python
results = rag.search("protein structure prediction", top_k=5)

for r in results:
    print(f"Title: {r['title']}")
    print(f"Section: {r['section']}")
    
    if r['image_paths']:
        print(f"Associated figures: {r['figure_ids']}")
        for img_path in r['image_paths']:
            # Display or process the image
            print(f"  Image: {img_path}")
```

## Choosing Between Backends

| Feature | ChromaDB | Pinecone |
|---------|----------|----------|
| Setup | No external service needed | Requires API key |
| Persistence | Local disk | Cloud-managed |
| Scalability | Limited by local resources | Highly scalable |
| Cost | Free | Pay per usage |
| Latency | Very fast (local) | Network latency |
| Sharing | Single machine | Multi-service access |
| Use case | Development, small datasets | Production, large datasets |

## Integration with Agent

The agent automatically selects the RAG backend based on the `RAG_BACKEND` environment variable:

```python
# In agent.py
from agent.rag.chroma_rag import ChromaPDFRAG
from agent.rag.pinecone_rag import PineconePDFRAG

def create_rag_backend():
    backend = os.getenv("RAG_BACKEND", "chroma").lower()
    if backend == "pinecone":
        return PineconePDFRAG(...)
    else:
        return ChromaPDFRAG(...)
```

The `search_research_papers` tool wraps whichever backend is configured, so the agent doesn't need to know which one is in use.

## Files

- `pdf_processing.py` - Shared PDF parsing, image extraction, chunking, and embedding utilities
- `chroma_rag.py` - ChromaDB-backed RAG with hybrid search and image-chunk linking
- `pinecone_rag.py` - Pinecone-backed RAG with optional hybrid search and image-chunk linking
- `rag_cli.py` - Command-line interface for indexing and search

## Data Directory Structure

After indexing, your data directory will look like:

```
data/
├── papers/                    # Source PDFs
│   ├── paper1.pdf
│   └── paper2.pdf
├── chroma_db/                 # ChromaDB vector store
├── bm25/                      # BM25 index (pickle files)
└── pdf_images/                # Extracted images
    ├── paper1/
    │   ├── page_0_img_0.png
    │   ├── page_1_img_0.png
    │   └── page_3_img_0.png
    └── paper2/
        └── page_2_img_0.png
```
