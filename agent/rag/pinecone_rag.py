#!/usr/bin/env python3
"""
Pinecone RAG Backend
~~~~~~~~~~~~~~~~~~~~
RAG system using Pinecone for vector storage with optional hybrid search (Vector + BM25).
Uses shared PDF processing utilities from pdf_processing.py.
Supports figure/table detection and image-chunk linking.
"""

import os
import pickle
import json
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np

from pinecone import Pinecone, ServerlessSpec
from rank_bm25 import BM25Okapi

from agent.rag.pdf_processing import (
    get_embeddings,
    process_pdf,
    list_pdfs,
    DEFAULT_IMAGE_OUTPUT_ROOT,
)


class PineconePDFRAG:
    """
    RAG system using Pinecone for vector storage with optional BM25 hybrid search.
    """
    
    def __init__(
        self,
        pdf_directory: str = "./data/papers",
        bm25_persist_directory: str = "./data/bm25_pinecone",
        embedding_service_url: str = "http://localhost:8011/embed",
        pinecone_api_key: Optional[str] = None,
        pinecone_index_name: str = "paper-rag-index",
        pinecone_namespace: str = "papers",
        pinecone_cloud: str = "aws",
        pinecone_region: str = "us-east-1",
        dimension: int = 768,  # PubMedBERT embedding dimension
        metric: str = "cosine",
        use_hybrid_search: bool = True,
        image_output_root: str = DEFAULT_IMAGE_OUTPUT_ROOT,
        extract_images: bool = True,
    ):
        self.pdf_directory = pdf_directory
        self.bm25_persist_directory = bm25_persist_directory
        self.embedding_service_url = embedding_service_url
        self.pinecone_index_name = pinecone_index_name
        self.pinecone_namespace = pinecone_namespace
        self.dimension = dimension
        self.use_hybrid_search = use_hybrid_search
        self.image_output_root = image_output_root
        self.extract_images = extract_images
        
        # Create directories
        os.makedirs(pdf_directory, exist_ok=True)
        os.makedirs(bm25_persist_directory, exist_ok=True)
        if extract_images:
            os.makedirs(image_output_root, exist_ok=True)
        
        print(f"Using embedding service at {embedding_service_url}")
        
        # Initialize Pinecone client
        api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found. Set it via env var or pass pinecone_api_key.")
        
        self.pc = Pinecone(api_key=api_key)
        
        # Create or connect to index
        self._ensure_index_exists(pinecone_cloud, pinecone_region, metric)
        self.index = self.pc.Index(self.pinecone_index_name)
        
        # Initialize BM25 for hybrid search (optional)
        self.bm25 = None
        self.bm25_doc_ids = []
        self.bm25_documents = []  # Store documents for BM25
        if self.use_hybrid_search:
            self._load_bm25()
        
        # Count PDFs
        self.paper_count = len(list_pdfs(pdf_directory))

        print(f"PineconePDFRAG initialized:")
        print(f"  PDF directory: {pdf_directory}")
        print(f"  Num PDFs: {self.paper_count}")
        print(f"  Pinecone index: {pinecone_index_name}")
        print(f"  Namespace: {pinecone_namespace}")
        print(f"  Hybrid search: {use_hybrid_search}")

    def _ensure_index_exists(self, cloud: str, region: str, metric: str):
        """Create Pinecone index if it doesn't exist"""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.pinecone_index_name not in existing_indexes:
            print(f"Creating Pinecone index: {self.pinecone_index_name}")
            self.pc.create_index(
                name=self.pinecone_index_name,
                dimension=self.dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region)
            )
            print(f"  Index created successfully")
        else:
            print(f"Using existing Pinecone index: {self.pinecone_index_name}")

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from microservice"""
        return get_embeddings(texts, self.embedding_service_url)

    def _load_bm25(self, force_rebuild: bool = False):
        """Load or rebuild BM25 index"""
        bm25_path = os.path.join(self.bm25_persist_directory, "bm25_pinecone.pkl")
        
        if not force_rebuild and os.path.exists(bm25_path):
            print("Loading BM25 index from disk...")
            try:
                with open(bm25_path, 'rb') as f:
                    data = pickle.load(f)
                    self.bm25 = data['bm25']
                    self.bm25_doc_ids = data['doc_ids']
                    self.bm25_documents = data.get('documents', [])
                print(f"  BM25 loaded with {len(self.bm25_doc_ids)} documents.")
                return
            except Exception as e:
                print(f"  ❌ Failed to load BM25 pickle: {e}")
        
        # If no local BM25 exists and we have documents in Pinecone, we can't easily rebuild
        # BM25 without fetching all vectors. For now, we'll build incrementally.
        print("BM25 index will be built incrementally during indexing.")

    def _save_bm25(self):
        """Save BM25 index to disk"""
        bm25_path = os.path.join(self.bm25_persist_directory, "bm25_pinecone.pkl")
        try:
            with open(bm25_path, 'wb') as f:
                pickle.dump({
                    'bm25': self.bm25,
                    'doc_ids': self.bm25_doc_ids,
                    'documents': self.bm25_documents
                }, f)
            print("  BM25 index saved to disk.")
        except Exception as e:
            print(f"  ❌ Failed to save BM25: {e}")

    def _rebuild_bm25(self):
        """Rebuild BM25 index from stored documents"""
        if not self.bm25_documents:
            print("  No documents to build BM25 index.")
            return
        
        tokenized_corpus = [doc.split() for doc in self.bm25_documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"  BM25 rebuilt with {len(self.bm25_documents)} documents.")
        self._save_bm25()

    def index_pdf(self, pdf_path: str, reindex: bool = False) -> int:
        """
        Index a single PDF into Pinecone with optional image extraction and figure-chunk linking.

        Args:
            pdf_path: Path to the PDF file
            reindex: If True, re-index even if already present

        Returns:
            Number of chunks indexed
        """
        paper_id = Path(pdf_path).stem

        # Check if already indexed
        if not reindex:
            try:
                sample_id = f"{paper_id}_chunk_0"
                fetch_result = self.index.fetch(ids=[sample_id], namespace=self.pinecone_namespace)
                if fetch_result.vectors and sample_id in fetch_result.vectors:
                    print(f"\nProcessing: {Path(pdf_path).name}")
                    print(f"  Skipping (already indexed)")
                    return 0
            except Exception:
                pass

        result = process_pdf(pdf_path, self.embedding_service_url, self.image_output_root, self.extract_images)
        metadata, chunks, embeddings = result["metadata"], result["chunks"], result["embeddings"]

        if not chunks:
            return 0

        vectors = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{paper_id}_chunk_{idx}"

            content = chunk["content"]
            if len(content) > 40000:  # Pinecone metadata limit is ~40KB
                content = content[:40000]

            figure_ids = chunk.get("figure_ids", [])
            table_ids = chunk.get("table_ids", [])
            image_paths = chunk.get("image_paths", [])

            if len(json.dumps(image_paths)) > 1000:
                image_paths = image_paths[:5]

            vector_metadata = {
                "content": content,
                "title": metadata['title'][:500],
                "author": metadata['author'][:200],
                "filename": metadata['filename'],
                "section": chunk["section"][:200],
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "has_table": chunk.get("has_table", False),
                "has_figure": chunk.get("has_figure", False),
                "figure_ids": json.dumps(figure_ids),
                "table_ids": json.dumps(table_ids),
                "image_paths": json.dumps(image_paths),
            }

            vectors.append({
                "id": chunk_id,
                "values": embedding,
                "metadata": vector_metadata
            })

            if self.use_hybrid_search:
                self.bm25_doc_ids.append(chunk_id)
                self.bm25_documents.append(chunk["content"])

        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=self.pinecone_namespace)

        print(f"  Upserted {len(vectors)} vectors to Pinecone")

        if self.use_hybrid_search:
            self._rebuild_bm25()

        return len(chunks)

    def index_local_pdfs(self, reindex: bool = False):
        """Index all PDFs in directory"""
        pdf_files = list_pdfs(self.pdf_directory)
        
        if not pdf_files:
            print(f"No PDFs found in {self.pdf_directory}")
            return
        self.paper_count = len(pdf_files)

        print(f"Found {len(pdf_files)} PDFs")
        
        if reindex:
            print("Clearing Pinecone namespace...")
            try:
                self.index.delete(delete_all=True, namespace=self.pinecone_namespace)
            except Exception as e:
                print(f"  Warning: Could not clear namespace: {e}")
            
            # Reset BM25
            self.bm25 = None
            self.bm25_doc_ids = []
            self.bm25_documents = []

        total_chunks = 0
        for pdf_path in pdf_files:
            try:
                total_chunks += self.index_pdf(str(pdf_path), reindex=reindex)
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
        
        # Rebuild BM25 after all indexing
        if self.use_hybrid_search:
            self._rebuild_bm25()
        
        print(f"Indexing complete. Total chunks: {total_chunks}")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search using Pinecone vector search with optional BM25 hybrid fusion"""
        
        # 1. Vector Search via Pinecone
        query_embedding = self._get_embeddings([query])[0]
        
        pinecone_results = self.index.query(
            vector=query_embedding,
            top_k=top_k * 2,  # Fetch more for reranking
            include_metadata=True,
            namespace=self.pinecone_namespace
        )
        
        # 2. BM25 Search (if enabled)
        bm25_results = []
        if self.use_hybrid_search and self.bm25:
            tokenized_query = query.split()
            scores = self.bm25.get_scores(tokenized_query)
            # Get top N indices
            top_n_indices = np.argsort(scores)[::-1][:top_k * 2]
            
            for idx in top_n_indices:
                if idx < len(self.bm25_doc_ids) and scores[idx] > 0:
                    doc_id = self.bm25_doc_ids[idx]
                    bm25_results.append(doc_id)
        
        # 3. Reciprocal Rank Fusion (RRF) if hybrid, else just use Pinecone scores
        rank_scores = {}
        
        # Process Pinecone Vector Results
        if pinecone_results.matches:
            for rank, match in enumerate(pinecone_results.matches):
                doc_id = match.id
                if doc_id not in rank_scores:
                    rank_scores[doc_id] = 0
                rank_scores[doc_id] += 1 / (60 + rank)
        
        # Process BM25 Results (if hybrid)
        if self.use_hybrid_search:
            for rank, doc_id in enumerate(bm25_results):
                if doc_id not in rank_scores:
                    rank_scores[doc_id] = 0
                rank_scores[doc_id] += 1 / (60 + rank)
        
        # Sort by Score
        sorted_ids = sorted(rank_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        if not sorted_ids:
            return []
        
        # Build results from Pinecone metadata
        # Create a lookup from Pinecone results
        id_to_match = {match.id: match for match in pinecone_results.matches}
        
        final_results = []
        for doc_id, score in sorted_ids:
            if doc_id in id_to_match:
                match = id_to_match[doc_id]
                meta = match.metadata or {}
                content = meta.get('content', '')
                
                # Parse JSON-encoded list fields
                figure_ids = []
                table_ids = []
                image_paths = []
                try:
                    figure_ids = json.loads(meta.get('figure_ids', '[]'))
                except (json.JSONDecodeError, TypeError):
                    pass
                try:
                    table_ids = json.loads(meta.get('table_ids', '[]'))
                except (json.JSONDecodeError, TypeError):
                    pass
                try:
                    image_paths = json.loads(meta.get('image_paths', '[]'))
                except (json.JSONDecodeError, TypeError):
                    pass
                
                final_results.append({
                    "title": meta.get('title', ''),
                    "section": meta.get('section', ''),
                    "content": content,
                    "content_preview": content[:200] + "..." if len(content) > 200 else content,
                    "score": score,
                    "filename": meta.get('filename', ''),
                    "pinecone_score": match.score,
                    "has_table": meta.get('has_table', False),
                    "has_figure": meta.get('has_figure', False),
                    "figure_ids": figure_ids,
                    "table_ids": table_ids,
                    "image_paths": image_paths,
                })
            else:
                # Doc from BM25 but not in Pinecone top results - need to fetch
                try:
                    fetch_result = self.index.fetch(ids=[doc_id], namespace=self.pinecone_namespace)
                    if fetch_result.vectors and doc_id in fetch_result.vectors:
                        vec = fetch_result.vectors[doc_id]
                        meta = vec.metadata or {}
                        content = meta.get('content', '')
                        
                        # Parse JSON-encoded list fields
                        figure_ids = []
                        table_ids = []
                        image_paths = []
                        try:
                            figure_ids = json.loads(meta.get('figure_ids', '[]'))
                        except (json.JSONDecodeError, TypeError):
                            pass
                        try:
                            table_ids = json.loads(meta.get('table_ids', '[]'))
                        except (json.JSONDecodeError, TypeError):
                            pass
                        try:
                            image_paths = json.loads(meta.get('image_paths', '[]'))
                        except (json.JSONDecodeError, TypeError):
                            pass
                        
                        final_results.append({
                            "title": meta.get('title', ''),
                            "section": meta.get('section', ''),
                            "content": content,
                            "content_preview": content[:200] + "..." if len(content) > 200 else content,
                            "score": score,
                            "filename": meta.get('filename', ''),
                            "pinecone_score": None,
                            "has_table": meta.get('has_table', False),
                            "has_figure": meta.get('has_figure', False),
                            "figure_ids": figure_ids,
                            "table_ids": table_ids,
                            "image_paths": image_paths,
                        })
                except Exception:
                    pass  # Skip if fetch fails
                
        return final_results

    def list_indexed_papers(self) -> Dict:
        """Get list of indexed papers with metadata"""
        try:
            # Get index stats to see namespaces
            stats = self.index.describe_index_stats()
            namespace_stats = stats.namespaces.get(self.pinecone_namespace, {})
            vector_count = namespace_stats.vector_count if hasattr(namespace_stats, 'vector_count') else 0
            
            # We can't easily list all unique papers without fetching all vectors
            # Return what we can from local BM25 data
            papers = {}
            for doc_id in self.bm25_doc_ids:
                # Extract paper name from chunk ID (format: papername_chunk_N)
                parts = doc_id.rsplit('_chunk_', 1)
                if len(parts) == 2:
                    paper_name = parts[0]
                    if paper_name not in papers:
                        papers[paper_name] = {"chunks": 0}
                    papers[paper_name]['chunks'] += 1
            
            return papers
        except Exception as e:
            print(f"Error listing papers: {e}")
            return {}

    def get_statistics(self) -> Dict:
        """Get system statistics"""
        try:
            stats = self.index.describe_index_stats()
            namespace_stats = stats.namespaces.get(self.pinecone_namespace, {})
            vector_count = namespace_stats.vector_count if hasattr(namespace_stats, 'vector_count') else 0
        except Exception:
            vector_count = 0
        
        papers = self.list_indexed_papers()
        
        return {
            "total_papers": len(papers),
            "total_chunks": vector_count,
            "pdf_directory": self.pdf_directory,
            "backend": "pinecone",
            "index_name": self.pinecone_index_name,
            "namespace": self.pinecone_namespace,
            "hybrid_search": self.use_hybrid_search,
            "papers": papers
        }

    def search_with_tables_or_figures(
        self, 
        query: str = "", 
        only_tables: bool = False, 
        only_figures: bool = False, 
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search for documents containing tables or figures
        
        Args:
            query: Search query (empty to get all table/figure documents)
            only_tables: Only return documents with tables
            only_figures: Only return documents with figures
            top_k: Number of results to return
        """
        # Build filter for Pinecone
        filter_dict = {}
        if only_tables:
            filter_dict["has_table"] = {"$eq": True}
        elif only_figures:
            filter_dict["has_figure"] = {"$eq": True}
        else:
            filter_dict = {
                "$or": [
                    {"has_table": {"$eq": True}},
                    {"has_figure": {"$eq": True}}
                ]
            }
        
        if query:
            # Vector search with filter
            query_embedding = self._get_embeddings([query])[0]
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=self.pinecone_namespace,
                filter=filter_dict
            )
            
            final_results = []
            for match in results.matches:
                meta = match.metadata or {}
                content = meta.get('content', '')
                
                # Parse JSON-encoded list fields
                figure_ids = []
                table_ids = []
                image_paths = []
                try:
                    figure_ids = json.loads(meta.get('figure_ids', '[]'))
                except (json.JSONDecodeError, TypeError):
                    pass
                try:
                    table_ids = json.loads(meta.get('table_ids', '[]'))
                except (json.JSONDecodeError, TypeError):
                    pass
                try:
                    image_paths = json.loads(meta.get('image_paths', '[]'))
                except (json.JSONDecodeError, TypeError):
                    pass
                
                final_results.append({
                    "title": meta.get('title', ''),
                    "section": meta.get('section', ''),
                    "content": content,
                    "content_preview": content[:200] + "..." if len(content) > 200 else content,
                    "has_table": meta.get('has_table', False),
                    "has_figure": meta.get('has_figure', False),
                    "filename": meta.get('filename', ''),
                    "score": match.score,
                    "figure_ids": figure_ids,
                    "table_ids": table_ids,
                    "image_paths": image_paths,
                })
            return final_results
        else:
            # No query - just return results matching the filter
            # Use a zero vector to get random results matching filter
            zero_vector = [0.0] * self.dimension
            results = self.index.query(
                vector=zero_vector,
                top_k=top_k,
                include_metadata=True,
                namespace=self.pinecone_namespace,
                filter=filter_dict
            )
            
            final_results = []
            for match in results.matches:
                meta = match.metadata or {}
                content = meta.get('content', '')
                
                # Parse JSON-encoded list fields
                figure_ids = []
                table_ids = []
                image_paths = []
                try:
                    figure_ids = json.loads(meta.get('figure_ids', '[]'))
                except (json.JSONDecodeError, TypeError):
                    pass
                try:
                    table_ids = json.loads(meta.get('table_ids', '[]'))
                except (json.JSONDecodeError, TypeError):
                    pass
                try:
                    image_paths = json.loads(meta.get('image_paths', '[]'))
                except (json.JSONDecodeError, TypeError):
                    pass
                
                final_results.append({
                    "title": meta.get('title', ''),
                    "section": meta.get('section', ''),
                    "content": content,
                    "content_preview": content[:200] + "..." if len(content) > 200 else content,
                    "has_table": meta.get('has_table', False),
                    "has_figure": meta.get('has_figure', False),
                    "filename": meta.get('filename', ''),
                    "figure_ids": figure_ids,
                    "table_ids": table_ids,
                    "image_paths": image_paths,
                })
            return final_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="store_true", help="Index PDFs")
    parser.add_argument("--search", type=str, help="Search query")
    parser.add_argument("--reindex", action="store_true", help="Force reindex")
    parser.add_argument("--no-hybrid", action="store_true", help="Disable hybrid search (vector only)")
    args = parser.parse_args()
    
    rag = PineconePDFRAG(use_hybrid_search=not args.no_hybrid)
    
    if args.index:
        rag.index_local_pdfs(reindex=args.reindex)
    
    if args.search:
        results = rag.search(args.search)
        for i, res in enumerate(results, 1):
            print(f"\n{i}. {res['title']}")
            print(f"   Section: {res['section']}")
            print(f"   Score: {res['score']:.4f}")
            if res.get('pinecone_score'):
                print(f"   Pinecone Score: {res['pinecone_score']:.4f}")
            print(f"   {res['content_preview']}")
    
    if not args.index and not args.search:
        print("Use --index to index PDFs or --search 'query' to search.")
        print("Stats:", rag.get_statistics())
