"""
RAG Module
~~~~~~~~~~
Retrieval-Augmented Generation backends for the Protein Design Agent.

Supports two vector storage backends:
- ChromaDB (local, default)
- Pinecone (managed, cloud-based)

Both backends share the same PDF processing pipeline (pdf_processing.py)
and expose a compatible interface for search and indexing.
"""

from agent.rag.chroma_rag import ChromaPDFRAG
from agent.rag.pinecone_rag import PineconePDFRAG
from agent.rag.pdf_processing import (
    get_embeddings,
    extract_metadata_from_pdf,
    extract_text_as_markdown,
    extract_pdf_with_page_chunks,
    extract_images_from_pdf,
    chunk_text,
    chunk_text_with_pages,
    process_pdf,
    list_pdfs,
)

__all__ = [
    # RAG Backends
    "ChromaPDFRAG",
    "PineconePDFRAG",
    # PDF Processing utilities
    "get_embeddings",
    "extract_metadata_from_pdf",
    "extract_text_as_markdown",
    "extract_pdf_with_page_chunks",
    "extract_images_from_pdf",
    "chunk_text",
    "chunk_text_with_pages",
    "process_pdf",
    "list_pdfs",
]
