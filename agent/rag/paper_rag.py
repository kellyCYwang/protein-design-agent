#!/usr/bin/env python3
"""
Enhanced RAG system with Hybrid Search (Vector + BM25) and Markdown Parsing
Features:
- Markdown Parsing (using PyMuPDF4LLM)
- Hierarchical Chunking (Header -> Paragraph)
- Context Enrichment (Title + Section)
- Hybrid Search (Vector + BM25 with RRF)
"""

import os
import pickle
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
import requests

import pymupdf4llm
import chromadb
# from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

class LocalPDFRAG:
    """
    Advanced RAG system for local PDF directory
    """
    
    def __init__(
        self,
        pdf_directory: str = "./data/papers",
        vd_persist_directory: str = "./data/chroma_db",
        bm25_persist_directory: str = "./data/bm25",
        embedding_service_url: str = "http://localhost:8011/embed"
    ):
        self.pdf_directory = pdf_directory
        self.vd_persist_directory = vd_persist_directory
        self.bm25_persist_directory = bm25_persist_directory
        self.embedding_service_url = embedding_service_url
        
        # Create directories
        os.makedirs(pdf_directory, exist_ok=True)
        os.makedirs(vd_persist_directory, exist_ok=True)
        os.makedirs(bm25_persist_directory, exist_ok=True)
        
        # Initialize embedding model
        # Removed local loading: self.embedding_model = SentenceTransformer('neuml/pubmedbert-base-embeddings')
        print(f"Using embedding service at {embedding_service_url}")
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=vd_persist_directory)  
        self.collection = self.client.get_or_create_collection(
            name="local_papers_enhanced",
            metadata={"description": "Papers with Markdown parsing and Hybrid Search"}
        )
        
        # Initialize BM25
        self.bm25 = None
        self.bm25_doc_ids = []  # Map BM25 index to Chunk ID
        self._load_bm25()
        
        # Count PDFs
        self.paper_count = len(list(Path(pdf_directory).glob("*.pdf")))

        print(f"RAG initialized:")
        print(f"  PDF directory: {pdf_directory}")
        print(f"  Num PDFs: {self.paper_count}")
        print(f"  Indexed chunks: {self.collection.count()}")

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from microservice"""
        try:
            response = requests.post(
                self.embedding_service_url,
                json={"text": texts},
                timeout=30
            )
            response.raise_for_status()
            return response.json()["embeddings"]
        except Exception as e:
            print(f"❌ Embedding service failed: {e}")
            # Fallback or raise error? For now raise to ensure we know it failed
            raise RuntimeError(f"Failed to get embeddings: {e}")
    
    
    def _load_bm25(self, force_rebuild: bool = False):
        """Load or rebuild BM25 index"""
        bm25_path = os.path.join(self.bm25_persist_directory, "bm25.pkl")   
        
        if not force_rebuild and os.path.exists(bm25_path):
            print("Loading BM25 index from disk...")
            try:
                with open(bm25_path, 'rb') as f:
                    data = pickle.load(f)
                    self.bm25 = data['bm25']
                    self.bm25_doc_ids = data['doc_ids']
                print(f"  BM25 loaded with {len(self.bm25_doc_ids)} documents.")
                return
            except Exception as e:
                print(f"  ❌ Failed to load BM25 pickle: {e}")
        
        print("Building BM25 index from ChromaDB...")
        try:
            result = self.collection.get()
            documents = result['documents']
            ids = result['ids']
            
            if not documents:
                print("  Index is empty.")
                return

            tokenized_corpus = [doc.split() for doc in documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self.bm25_doc_ids = ids
            print(f"  BM25 built with {len(documents)} documents.")
            
            # Save to disk
            self._save_bm25()
            
        except Exception as e:
            print(f"  ❌ Failed to build BM25: {e}")

    def _save_bm25(self):
        """Save BM25 index to disk"""
        bm25_path = os.path.join(self.bm25_persist_directory, "bm25.pkl")   
        try:
            with open(bm25_path, 'wb') as f:
                pickle.dump({
                    'bm25': self.bm25,
                    'doc_ids': self.bm25_doc_ids
                }, f)
            print("  BM25 index saved to disk.")
        except Exception as e:
            print(f"  ❌ Failed to save BM25: {e}")

    def extract_metadata_from_pdf(self, pdf_path: str) -> Dict:
        """Extract metadata using PyMuPDF"""
        import fitz
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            title = metadata.get('title', '') or metadata.get('Title', '')
            if not title:
                title = Path(pdf_path).stem.replace('_', ' ').replace('-', ' ')
            author = metadata.get('author', '') or metadata.get('Author', 'Unknown')
            doc.close()
            return {"title": title, "author": author, "filename": Path(pdf_path).name}
        except:
            return {"title": Path(pdf_path).stem, "author": "Unknown", "filename": Path(pdf_path).name}

    def extract_text_as_markdown(self, pdf_path: str) -> str:
        """Convert PDF to Markdown using PyMuPDF4LLM with enhanced table/image handling"""
        try:
            print(f"  Parsing PDF to Markdown: {Path(pdf_path).name}")
            return pymupdf4llm.to_markdown(
                pdf_path,
                table_strategy="lines_strict",  # Best for scientific tables
                ignore_graphics=False,  # Ensure tables/graphics are processed
                ignore_images=False,  # Process images (though not embedded by default)
                force_text=True,  # Ensure text extraction even with image backgrounds
                show_progress=False  # Disable progress for cleaner output
            )
        except Exception as e:
            print(f"  ❌ Markdown conversion failed: {e}")
            return ""

    def chunk_text(self, text: str, title: str) -> List[Dict]:
        """
        Hierarchical Chunking:
        1. Split by Headers (#, ##)
        2. Split by Paragraphs (RecursiveCharacterTextSplitter)
        3. Enrich with Title + Section context
        """
        if not text:
            return []

        # 1. Split by Header
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(text)

        # 2. Split by Character (Paragraphs)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        final_chunks = []
        
        for split in md_header_splits:
            # Prepare Context
            header_context = []
            if "Header 1" in split.metadata:
                header_context.append(split.metadata["Header 1"])
            if "Header 2" in split.metadata:
                header_context.append(split.metadata["Header 2"])
            if "Header 3" in split.metadata:
                header_context.append(split.metadata["Header 3"])
            
            section_str = " > ".join(header_context) if header_context else "General"
            
            # Sub-split large sections
            sub_splits = text_splitter.split_text(split.page_content)
            
            for sub_split in sub_splits:
                # Enrichment
                enriched_content = f"Paper Title: {title} | Section: {section_str}\n\n{sub_split}"
                
                # Detect table/figure content
                content_lower = sub_split.lower()
                has_table = "table" in content_lower
                has_figure = "figure" in content_lower
                
                final_chunks.append({
                    "content": enriched_content,
                    "original_content": sub_split,
                    "section": section_str,
                    "metadata": split.metadata,
                    "has_table": has_table,
                    "has_figure": has_figure
                })
        
        return final_chunks

    def index_pdf(self, pdf_path: str, reindex: bool = False) -> int:
        """Index a single PDF"""
        print(f"\nProcessing: {Path(pdf_path).name}")
        
        # Check if already indexed (naive check by filename in metadata)
        if not reindex:
            existing = self.collection.get(where={"filename": Path(pdf_path).name})
            if existing['ids']:
                print(f"  Skipping (already indexed)")
                return 0

        metadata = self.extract_metadata_from_pdf(pdf_path)
        print(f"  Title: {metadata['title'][:60]}...")

        # Extract Markdown
        md_text = self.extract_text_as_markdown(pdf_path)
        if not md_text:
            return 0
        
        # Chunk
        chunks = self.chunk_text(md_text, metadata['title'])
        print(f"  Created {len(chunks)} chunks")
        
        if not chunks:
            return 0

        # Prepare for Chroma
        documents = []
        embeddings = []
        metadatas = []
        ids = []
        
        paper_id = Path(pdf_path).stem
        
        # Batch embedding generation
        texts_to_embed = [chunk["content"] for chunk in chunks]
        embeddings_list = self._get_embeddings(texts_to_embed)
        
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings_list)):
            chunk_id = f"{paper_id}_chunk_{idx}"
            
            documents.append(chunk["content"])
            embeddings.append(embedding)
            ids.append(chunk_id)
            
            meta = {
                "title": metadata['title'][:500],
                "author": metadata['author'][:200],
                "filename": metadata['filename'],
                "section": chunk["section"][:200],
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "has_table": chunk.get("has_table", False),
                "has_figure": chunk.get("has_figure", False)
            }
            metadatas.append(meta)

        # Add to Chroma
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        # Update in-memory BM25
        # (For efficiency, we might just rebuild fully at end of batch, but for single file:)
        if self.bm25 is None:
            self._load_bm25()
        else:
            # Incremental update is hard with BM25Okapi class, so we might need to rebuild
            # or just append if we implement a custom one. 
            # For simplicity, we'll rebuild BM25 after indexing a batch of files.
            pass 

        return len(chunks)

    def index_local_pdfs(self, reindex: bool = False):
        """Index all PDFs in directory"""
        pdf_files = list(Path(self.pdf_directory).glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDFs found in {self.pdf_directory}")
            return
        self.paper_count = len(pdf_files)

        print(f"Found {len(pdf_files)} PDFs")
        
        if reindex:
            print("Clearing index...")
            self.client.delete_collection("local_papers_enhanced")
            self.collection = self.client.get_or_create_collection("local_papers_enhanced")
            self.bm25 = None
            self.bm25_doc_ids = []

        total_chunks = 0
        for pdf_path in pdf_files:
            try:
                total_chunks += self.index_pdf(str(pdf_path), reindex=reindex)
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
        
        # Rebuild BM25 after all indexing
        self._load_bm25(force_rebuild=True)
        print(f"Indexing complete. Total chunks: {total_chunks}")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Hybrid Search (Vector + BM25) with RRF"""
        
        # 1. Vector Search
        query_embedding = self._get_embeddings([query])[0]
        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2  # Fetch more for reranking
        )
        
        # 2. BM25 Search
        bm25_results = []
        if self.bm25:
            tokenized_query = query.split()
            scores = self.bm25.get_scores(tokenized_query)
            # Get top N indices
            top_n_indices = np.argsort(scores)[::-1][:top_k * 2]
            
            for idx in top_n_indices:
                if scores[idx] > 0: # Only relevant
                    doc_id = self.bm25_doc_ids[idx]
                    bm25_results.append(doc_id)
        
        # 3. Reciprocal Rank Fusion (RRF)
        rank_scores = {}
        
        # Process Vector Results
        if vector_results['ids']:
            for rank, doc_id in enumerate(vector_results['ids'][0]):
                if doc_id not in rank_scores:
                    rank_scores[doc_id] = 0
                rank_scores[doc_id] += 1 / (60 + rank)
        
        # Process BM25 Results
        for rank, doc_id in enumerate(bm25_results):
            if doc_id not in rank_scores:
                rank_scores[doc_id] = 0
            rank_scores[doc_id] += 1 / (60 + rank)
        
        # Sort by Score
        sorted_ids = sorted(rank_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Fetch final details
        final_results = []
        target_ids = [doc_id for doc_id, _ in sorted_ids]
        
        if not target_ids:
            return []

        # Get documents for target IDs
        # Chroma .get() supports filtering by IDs
        docs_data = self.collection.get(ids=target_ids, include=['documents', 'metadatas'])
        
        # Map back to result format
        # Note: Chroma get() results might not be in request order
        id_to_data = {
            id_: (doc, meta) 
            for id_, doc, meta in zip(docs_data['ids'], docs_data['documents'], docs_data['metadatas'])
        }
        
        for doc_id in target_ids:
            if doc_id in id_to_data:
                doc, meta = id_to_data[doc_id]
                final_results.append({
                    "title": meta.get('title', ''),
                    "section": meta.get('section', ''),
                    "content": doc,
                    "content_preview": doc[:200] + "...",
                    "score": rank_scores[doc_id],
                    "filename": meta.get('filename', '')
                })
                
        return final_results

    def list_indexed_papers(self) -> Dict:
        """Get list of indexed papers with metadata"""
        try:
            # Get all documents metadata
            result = self.collection.get(include=['metadatas'])
            metadatas = result['metadatas']
            
            papers = {}
            for meta in metadatas:
                filename = meta.get('filename', 'Unknown')
                if filename not in papers:
                    papers[filename] = {
                        "title": meta.get('title', 'Unknown'),
                        "author": meta.get('author', 'Unknown'),
                        "chunks": 0
                    }
                papers[filename]['chunks'] += 1
            
            return papers
        except Exception as e:
            print(f"Error listing papers: {e}")
            return {}

    def get_statistics(self) -> Dict:
        """Get system statistics"""
        papers = self.list_indexed_papers()
        return {
            "total_papers": len(papers),
            "total_chunks": self.collection.count(),
            "pdf_directory": self.pdf_directory,
            "papers": papers
        }

    def search_with_tables_or_figures(self, query: str = "", only_tables: bool = False, only_figures: bool = False, top_k: int = 5) -> List[Dict]:
        """
        Search for documents containing tables or figures
        
        Args:
            query: Search query (empty to get all table/figure documents)
            only_tables: Only return documents with tables
            only_figures: Only return documents with figures
            top_k: Number of results to return
        """
        # Build filter based on parameters
        filter_conditions = {}
        
        if only_tables:
            filter_conditions["has_table"] = True
        elif only_figures:
            filter_conditions["has_figure"] = True
        else:
            # Both tables and figures
            filter_conditions = {
                "$or": [
                    {"has_table": True},
                    {"has_figure": True}
                ]
            }
        
        # Perform search
        if query:
            # Use hybrid search with filter
            results = self.search(query, top_k * 2)
            # Filter results based on table/figure criteria
            filtered_results = []
            for result in results:
                has_table = result.get("has_table", False)
                has_figure = result.get("has_figure", False)
                
                if (only_tables and has_table) or (only_figures and has_figure) or (not only_tables and not only_figures and (has_table or has_figure)):
                    filtered_results.append(result)
                    if len(filtered_results) >= top_k:
                        break
            return filtered_results[:top_k]
        else:
            # Get all documents with tables/figures
            result = self.collection.get(where=filter_conditions, limit=top_k)
            
            final_results = []
            for doc, meta in zip(result['documents'], result['metadatas']):
                final_results.append({
                    "title": meta.get('title', ''),
                    "section": meta.get('section', ''),
                    "content": doc,
                    "content_preview": doc[:200] + "...",
                    "has_table": meta.get('has_table', False),
                    "has_figure": meta.get('has_figure', False),
                    "filename": meta.get('filename', '')
                })
            
            return final_results

if __name__ == "__main__":
    rag = LocalPDFRAG()
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="store_true", help="Index PDFs")
    parser.add_argument("--search", type=str, help="Search query")
    parser.add_argument("--reindex", action="store_true", help="Force reindex")
    args = parser.parse_args()
    
    if args.index:
        rag.index_local_pdfs(reindex=args.reindex)
    
    if args.search:
        results = rag.search(args.search)
        for i, res in enumerate(results, 1):
            print(f"\n{i}. {res['title']}")
            print(f"   Section: {res['section']}")
            print(f"   Score: {res['score']:.4f}")
            print(f"   {res['content_preview']}")
    
    if not args.index and not args.search:
        print("Use --index to index PDFs or --search 'query' to search.")
