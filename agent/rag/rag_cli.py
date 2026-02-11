#!/usr/bin/env python3
"""
RAG CLI
~~~~~~~
Command-line interface for indexing PDFs and searching with either
ChromaDB or Pinecone backend.

Usage:
    python -m agent.rag.rag_cli --backend chroma --index
    python -m agent.rag.rag_cli --backend pinecone --search "RFDiffusion architecture"
    python -m agent.rag.rag_cli --stats
"""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def get_backend(backend_name: str, **kwargs):
    """
    Get the appropriate RAG backend instance.
    
    Args:
        backend_name: 'chroma' or 'pinecone'
        **kwargs: Additional arguments passed to the backend constructor
        
    Returns:
        ChromaPDFRAG or PineconePDFRAG instance
    """
    if backend_name.lower() == "pinecone":
        from agent.rag.pinecone_rag import PineconePDFRAG
        return PineconePDFRAG(
            pdf_directory=kwargs.get("pdf_dir", os.getenv("RAG_PDF_DIR", "./data/papers")),
            bm25_persist_directory=kwargs.get("bm25_dir", os.getenv("RAG_BM25_DIR", "./data/bm25_pinecone")),
            embedding_service_url=kwargs.get("embedding_url", os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8011/embed")),
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            pinecone_index_name=kwargs.get("index_name", os.getenv("PINECONE_INDEX_NAME", "paper-rag-index")),
            pinecone_namespace=kwargs.get("namespace", os.getenv("PINECONE_NAMESPACE", "papers")),
            use_hybrid_search=kwargs.get("hybrid", True),
            extract_images=False,
        )
    else:
        from agent.rag.chroma_rag import ChromaPDFRAG
        return ChromaPDFRAG(
            pdf_directory=kwargs.get("pdf_dir", os.getenv("RAG_PDF_DIR", "./data/papers")),
            vd_persist_directory=kwargs.get("chroma_dir", os.getenv("RAG_CHROMA_DIR", "./data/chroma_db")),
            bm25_persist_directory=kwargs.get("bm25_dir", os.getenv("RAG_BM25_DIR", "./data/bm25")),
            embedding_service_url=kwargs.get("embedding_url", os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8011/embed")),
            extract_images=False,
        )


def main():
    parser = argparse.ArgumentParser(
        description="RAG CLI - Index and search PDFs with Chroma or Pinecone backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index PDFs with Chroma backend
  python -m agent.rag.rag_cli --backend chroma --index

  # Force reindex with Pinecone
  python -m agent.rag.rag_cli --backend pinecone --index --reindex

  # Search with Chroma
  python -m agent.rag.rag_cli --backend chroma --search "RFDiffusion architecture"

  # Get statistics
  python -m agent.rag.rag_cli --backend pinecone --stats

Environment Variables:
  RAG_BACKEND          Default backend (chroma or pinecone)
  RAG_PDF_DIR          PDF directory (default: ./data/papers)
  RAG_CHROMA_DIR       Chroma persist directory (default: ./data/chroma_db)
  RAG_BM25_DIR         BM25 persist directory (default: ./data/bm25)
  PINECONE_API_KEY     Pinecone API key (required for pinecone backend)
  PINECONE_INDEX_NAME  Pinecone index name (default: paper-rag-index)
  EMBEDDING_SERVICE_URL Embedding service URL (default: http://localhost:8011/embed)
        """
    )
    
    # Backend selection
    parser.add_argument(
        "--backend", "-b",
        choices=["chroma", "pinecone"],
        default=os.getenv("RAG_BACKEND", "chroma"),
        help="RAG backend to use (default: from RAG_BACKEND env var or 'chroma')"
    )
    
    # Actions (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--index", "-i",
        action="store_true",
        help="Index all PDFs in the configured directory"
    )
    action_group.add_argument(
        "--search", "-s",
        type=str,
        metavar="QUERY",
        help="Search for a query"
    )
    action_group.add_argument(
        "--stats",
        action="store_true",
        help="Show index statistics"
    )
    action_group.add_argument(
        "--list-papers",
        action="store_true",
        help="List all indexed papers"
    )
    
    # Options
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force reindex (clear existing index first)"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of search results to return (default: 5)"
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        help="Override PDF directory"
    )
    parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="Disable hybrid search (vector only, Pinecone backend)"
    )
    
    args = parser.parse_args()
    
    # Build kwargs for backend construction
    kwargs = {}
    if args.pdf_dir:
        kwargs["pdf_dir"] = args.pdf_dir
    if args.no_hybrid:
        kwargs["hybrid"] = False
    
    # Get backend
    print(f"Initializing {args.backend} backend...")
    try:
        rag = get_backend(args.backend, **kwargs)
    except Exception as e:
        print(f"❌ Failed to initialize {args.backend} backend: {e}")
        sys.exit(1)
    
    # Execute action
    if args.index:
        print(f"\n{'='*60}")
        print(f"INDEXING PDFs")
        print(f"{'='*60}")
        rag.index_local_pdfs(reindex=args.reindex)
        
    elif args.search:
        print(f"\n{'='*60}")
        print(f"SEARCH: {args.search}")
        print(f"{'='*60}")
        results = rag.search(args.search, top_k=args.top_k)
        
        if not results:
            print("No results found.")
        else:
            for i, res in enumerate(results, 1):
                print(f"\n{i}. {res.get('title', 'Unknown')}")
                print(f"   Section: {res.get('section', 'N/A')}")
                print(f"   File: {res.get('filename', 'N/A')}")
                score = res.get('score')
                print(f"   Score: {score:.4f}" if isinstance(score, (int, float)) else f"   Score: {score}")
                preview = res.get('content_preview', res.get('content', '')[:200] + "...")
                print(f"   Preview: {preview}")
                
    elif args.stats:
        print(f"\n{'='*60}")
        print(f"INDEX STATISTICS")
        print(f"{'='*60}")
        stats = rag.get_statistics()
        for key, value in stats.items():
            if key != "papers":
                print(f"  {key}: {value}")
        
    elif args.list_papers:
        print(f"\n{'='*60}")
        print(f"INDEXED PAPERS")
        print(f"{'='*60}")
        papers = rag.list_indexed_papers()
        if not papers:
            print("No papers indexed.")
        else:
            for filename, info in papers.items():
                title = info.get("title", filename)
                chunks = info.get("chunks", 0)
                print(f"  - {title}")
                print(f"    File: {filename}, Chunks: {chunks}")
                
    else:
        # No action specified, show help
        parser.print_help()
        print("\n" + "="*60)
        print("CURRENT CONFIGURATION")
        print("="*60)
        print(f"  Backend: {args.backend}")
        stats = rag.get_statistics()
        print(f"  Total Papers: {stats.get('total_papers', 0)}")
        print(f"  Total Chunks: {stats.get('total_chunks', 0)}")
        print(f"  PDF Directory: {stats.get('pdf_directory', 'N/A')}")


if __name__ == "__main__":
    main()
