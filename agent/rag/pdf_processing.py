#!/usr/bin/env python3
"""
PDF Processing Module
~~~~~~~~~~~~~~~~~~~~~
Shared utilities for PDF parsing, chunking, embedding, and image extraction.
Uses Docling for deep-learning-based table/figure detection and structural chunking.
Used by both ChromaPDFRAG and PineconePDFRAG backends.
"""

import os
import re
from typing import List, Dict, Optional
from pathlib import Path

import requests
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.chunking import HierarchicalChunker
from docling_core.types.doc import TableItem, PictureItem


# ============================================
# CONSTANTS
# ============================================

FIGURE_PATTERN = re.compile(r"(?:Figure|Fig\.?)\s*(\d+[A-Za-z]?)", re.IGNORECASE)
TABLE_PATTERN = re.compile(r"Table\s*(\d+[A-Za-z]?)", re.IGNORECASE)

DEFAULT_IMAGE_OUTPUT_ROOT = os.getenv("RAG_IMAGE_DIR", "./data/pdf_images")
DEFAULT_EMBEDDING_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8011/embed")


# ============================================
# EMBEDDING SERVICE
# ============================================

def get_embeddings(
    texts: List[str],
    embedding_service_url: str = None
) -> List[List[float]]:
    """
    Get embeddings from microservice.

    Args:
        texts: List of text strings to embed
        embedding_service_url: URL of the embedding service

    Returns:
        List of embedding vectors

    Raises:
        RuntimeError: If embedding service fails
    """
    if embedding_service_url is None:
        embedding_service_url = DEFAULT_EMBEDDING_URL

    try:
        response = requests.post(
            embedding_service_url,
            json={"text": texts},
            timeout=30
        )
        response.raise_for_status()
        return response.json()["embeddings"]
    except Exception as e:
        print(f"❌ Embedding service failed: {e}")
        raise RuntimeError(f"Failed to get embeddings: {e}")


# ============================================
# DOCLING HELPERS
# ============================================

def _build_converter(extract_images: bool) -> DocumentConverter:
    """Build a Docling DocumentConverter with appropriate pipeline options."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.generate_picture_images = extract_images
    pipeline_options.images_scale = 2.0
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def _convert_with_docling(pdf_path: str, extract_images: bool):
    """Convert a PDF with Docling. Returns the DoclingDocument (single expensive parse)."""
    print(f"  Parsing PDF with Docling: {Path(pdf_path).name}")
    converter = _build_converter(extract_images)
    result = converter.convert(pdf_path)
    return result.document


# ============================================
# PDF METADATA EXTRACTION
# ============================================

def _extract_metadata_from_docling_doc(doc, pdf_path: str) -> Dict:
    """Extract metadata from an already-parsed DoclingDocument."""
    title = ""
    author = "Unknown"

    try:
        desc = getattr(doc, 'description', None)
        if desc is not None:
            title = getattr(desc, 'title', '') or ''
            authors = getattr(desc, 'authors', None)
            if authors:
                if isinstance(authors, list):
                    author = ", ".join(str(a) for a in authors) if authors else "Unknown"
                else:
                    author = str(authors)
    except AttributeError:
        pass

    if not title:
        title = Path(pdf_path).stem.replace('_', ' ').replace('-', ' ')

    return {"title": title, "author": author, "filename": Path(pdf_path).name}


def extract_metadata_from_pdf(pdf_path: str) -> Dict:
    """
    Extract metadata from a PDF file using Docling.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary with title, author, and filename
    """
    try:
        doc = _convert_with_docling(pdf_path, extract_images=False)
        return _extract_metadata_from_docling_doc(doc, pdf_path)
    except Exception:
        return {"title": Path(pdf_path).stem, "author": "Unknown", "filename": Path(pdf_path).name}


# ============================================
# MARKDOWN EXPORT (compat wrapper)
# ============================================

def extract_text_as_markdown(pdf_path: str) -> str:
    """
    Convert PDF to Markdown using Docling.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Markdown text extracted from PDF
    """
    try:
        print(f"  Parsing PDF to Markdown: {Path(pdf_path).name}")
        doc = _convert_with_docling(pdf_path, extract_images=False)
        return doc.export_to_markdown()
    except Exception as e:
        print(f"  ❌ Markdown conversion failed: {e}")
        return ""


# ============================================
# IMAGE EXTRACTION
# ============================================

def _extract_images_from_docling_doc(
    doc,
    pdf_path: str,
    output_root: str = DEFAULT_IMAGE_OUTPUT_ROOT,
    min_width: int = 50,
    min_height: int = 50
) -> List[Dict]:
    """Extract images from an already-parsed DoclingDocument."""
    paper_id = Path(pdf_path).stem
    paper_image_dir = Path(output_root) / paper_id
    paper_image_dir.mkdir(parents=True, exist_ok=True)

    images: List[Dict] = []

    try:
        pictures = list(doc.pictures) if hasattr(doc, 'pictures') else []
    except Exception:
        pictures = []

    for i, picture in enumerate(pictures):
        try:
            # Get PIL image
            pil_image = None
            try:
                pil_image = picture.image.pil_image
            except AttributeError:
                pass

            if pil_image is None:
                continue

            width, height = pil_image.size
            if width < min_width or height < min_height:
                continue

            # Save image
            image_filename = f"picture_{i}.png"
            image_path = paper_image_dir / image_filename
            pil_image.save(str(image_path))

            # Get caption
            caption = ""
            try:
                caption = picture.caption_text(doc)
            except AttributeError:
                try:
                    captions = picture.get_captions(doc)
                    caption = " ".join(str(c) for c in captions) if captions else ""
                except Exception:
                    pass

            # Extract figure_id from caption
            figure_id = None
            if caption:
                match = FIGURE_PATTERN.search(caption)
                if match:
                    figure_id = f"Figure {match.group(1)}"

            # Get page number (0-indexed)
            page = 0
            try:
                if picture.prov:
                    page = picture.prov[0].page_no - 1
            except (AttributeError, IndexError):
                pass

            self_ref = getattr(picture, 'self_ref', None)

            images.append({
                "page": page,
                "path": str(image_path),
                "figure_id": figure_id,
                "caption": caption,
                "self_ref": self_ref,
                "width": width,
                "height": height,
            })

        except Exception as e:
            print(f"    Warning: Could not extract picture {i}: {e}")
            continue

    print(f"  Extracted {len(images)} images to {paper_image_dir}")
    return images


def extract_images_from_pdf(
    pdf_path: str,
    output_root: str = DEFAULT_IMAGE_OUTPUT_ROOT,
    min_width: int = 50,
    min_height: int = 50
) -> List[Dict]:
    """
    Extract images from a PDF and save them to disk with metadata.

    Args:
        pdf_path: Path to the PDF file
        output_root: Root directory for saving extracted images
        min_width: Minimum image width to extract (skip tiny images)
        min_height: Minimum image height to extract

    Returns:
        List of image records with keys: page, path, figure_id, caption,
        self_ref, width, height
    """
    try:
        doc = _convert_with_docling(pdf_path, extract_images=True)
        return _extract_images_from_docling_doc(doc, pdf_path, output_root, min_width, min_height)
    except Exception as e:
        print(f"  ❌ Image extraction failed: {e}")
        return []


# ============================================
# TEXT CHUNKING HELPERS
# ============================================

def _has_markdown_table_syntax(text: str) -> bool:
    """Check if text contains markdown table syntax (| with --- separators)."""
    lines = text.split("\n")
    has_pipe_row = False
    has_separator = False

    for line in lines:
        stripped = line.strip()
        if "|" in stripped:
            has_pipe_row = True
            if re.match(r"^\|?\s*[-:]+\s*\|", stripped):
                has_separator = True

    return has_pipe_row and has_separator


def _has_image_markdown(text: str) -> bool:
    """Check if text contains markdown image syntax ![...](...) or similar."""
    return bool(re.search(r"!\[.*?\]\(.*?\)", text))


def _extract_figure_ids(text: str) -> List[str]:
    """Extract all figure references from text."""
    matches = FIGURE_PATTERN.findall(text)
    return [f"Figure {m}" for m in matches]


def _extract_table_ids(text: str) -> List[str]:
    """Extract all table references from text."""
    matches = TABLE_PATTERN.findall(text)
    return [f"Table {m}" for m in matches]


# ============================================
# DOCLING CHUNKING
# ============================================

def _docchunk_to_dict(chunk, title: str, images_by_ref: Dict, images_by_figure_id: Dict, doc=None) -> Dict:
    """Map a single DocChunk from HierarchicalChunker to our chunk dict format."""
    # Section from headings
    try:
        headings = chunk.meta.headings if chunk.meta.headings else []
        section = " > ".join(headings) if headings else "General"
    except AttributeError:
        section = "General"

    # Get page from first item's prov
    page = 0
    try:
        for item in chunk.meta.doc_items:
            if hasattr(item, 'prov') and item.prov:
                page = item.prov[0].page_no - 1  # 0-indexed
                break
    except (AttributeError, IndexError):
        pass

    # Build text: start with chunk.text, append table markdown for structural tables
    text = chunk.text
    has_table = False
    has_figure = False

    try:
        for item in chunk.meta.doc_items:
            if isinstance(item, TableItem):
                has_table = True
                try:
                    table_md = item.export_to_markdown(doc) if doc is not None else item.export_to_markdown()
                    if table_md and table_md not in text:
                        text = text + "\n\n" + table_md
                except Exception:
                    pass
            elif isinstance(item, PictureItem):
                has_figure = True
    except AttributeError:
        pass

    # Extract figure/table IDs from combined text
    figure_ids = _extract_figure_ids(text)
    table_ids = _extract_table_ids(text)

    # Refine detection flags
    has_table = has_table or bool(table_ids) or _has_markdown_table_syntax(text)
    has_figure = has_figure or bool(figure_ids) or _has_image_markdown(text)

    # Image linking — pass 1: structural match by PictureItem.self_ref
    image_paths: List[str] = []
    try:
        for item in chunk.meta.doc_items:
            if isinstance(item, PictureItem):
                self_ref = getattr(item, 'self_ref', None)
                if self_ref and self_ref in images_by_ref:
                    image_paths.append(images_by_ref[self_ref]["path"])
    except AttributeError:
        pass

    # Image linking — pass 2: fallback by figure_id
    if not image_paths:
        for fid in figure_ids:
            if fid in images_by_figure_id:
                image_paths.append(images_by_figure_id[fid]["path"])
            else:
                for img_fid, img in images_by_figure_id.items():
                    if fid in img_fid or img_fid in fid:
                        image_paths.append(img["path"])

    image_paths = sorted(set(image_paths))

    enriched_content = f"Paper Title: {title} | Section: {section} | Page: {page + 1}\n\n{text}"

    return {
        "content": enriched_content,
        "original_content": text,
        "section": section,
        "page": page,
        "has_table": has_table,
        "has_figure": has_figure,
        "figure_ids": figure_ids,
        "table_ids": table_ids,
        "image_paths": image_paths,
    }


def _chunk_docling_document(doc, title: str, images: List[Dict]) -> List[Dict]:
    """Chunk a DoclingDocument using HierarchicalChunker."""
    images_by_ref = {img["self_ref"]: img for img in images if img.get("self_ref")}
    images_by_figure_id = {img["figure_id"]: img for img in images if img.get("figure_id")}

    try:
        chunker = HierarchicalChunker(merge_peers=True)
    except TypeError:
        chunker = HierarchicalChunker()

    try:
        raw_chunks = list(chunker.chunk(doc))
    except Exception as e:
        print(f"  ⚠️ HierarchicalChunker failed: {e}")
        return []

    return [
        _docchunk_to_dict(c, title, images_by_ref, images_by_figure_id, doc=doc)
        for c in raw_chunks
        if c.text.strip()
    ]


# ============================================
# TEXT CHUNKING (compat shim — no longer called internally)
# ============================================

def chunk_text(text: str, title: str) -> List[Dict]:
    """
    Hierarchical chunking using LangChain splitters (compat shim).
    No longer called internally by process_pdf; kept for external callers.

    Args:
        text: Markdown text to chunk
        title: Paper title for context enrichment

    Returns:
        List of chunk dictionaries with content, metadata, flags, and reference IDs
    """
    if not text:
        return []

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    final_chunks = []

    for split in md_header_splits:
        header_context = []
        if "Header 1" in split.metadata:
            header_context.append(split.metadata["Header 1"])
        if "Header 2" in split.metadata:
            header_context.append(split.metadata["Header 2"])
        if "Header 3" in split.metadata:
            header_context.append(split.metadata["Header 3"])

        section_str = " > ".join(header_context) if header_context else "General"
        sub_splits = text_splitter.split_text(split.page_content)

        for sub_split in sub_splits:
            enriched_content = f"Paper Title: {title} | Section: {section_str}\n\n{sub_split}"
            figure_ids = _extract_figure_ids(sub_split)
            table_ids = _extract_table_ids(sub_split)
            has_table = bool(table_ids) or _has_markdown_table_syntax(sub_split)
            has_figure = bool(figure_ids) or _has_image_markdown(sub_split)

            final_chunks.append({
                "content": enriched_content,
                "original_content": sub_split,
                "section": section_str,
                "metadata": split.metadata,
                "has_table": has_table,
                "has_figure": has_figure,
                "figure_ids": figure_ids,
                "table_ids": table_ids,
            })

    return final_chunks


# ============================================
# BATCH PDF PROCESSING
# ============================================

def process_pdf(
    pdf_path: str,
    embedding_service_url: str = None,
    image_output_root: str = None,
    extract_images: bool = True,
    use_page_chunks: bool = True  # kept for signature compat; Docling is always structure-aware
) -> Dict:
    """
    Process a single PDF: parse with Docling, extract metadata, chunk by structure,
    embed, and optionally extract images with structural figure-chunk linking.

    Args:
        pdf_path: Path to the PDF file
        embedding_service_url: URL of the embedding service
        image_output_root: Root directory for extracted images
        extract_images: Whether to extract images from PDF
        use_page_chunks: No-op (kept for backward-compatible signature)

    Returns:
        Dictionary with:
        - metadata: PDF metadata (title, author, filename)
        - chunks: List of chunks with figure_ids, table_ids, image_paths, and page numbers
        - embeddings: List of embedding vectors for each chunk
        - images: List of extracted image records (if extract_images=True)
    """
    if embedding_service_url is None:
        embedding_service_url = DEFAULT_EMBEDDING_URL
    if image_output_root is None:
        image_output_root = DEFAULT_IMAGE_OUTPUT_ROOT

    print(f"\nProcessing: {Path(pdf_path).name}")

    # Single expensive parse — reused for all downstream steps
    doc = _convert_with_docling(pdf_path, extract_images=extract_images)

    metadata = _extract_metadata_from_docling_doc(doc, pdf_path)
    print(f"  Title: {metadata['title'][:60]}...")

    images: List[Dict] = []
    if extract_images:
        images = _extract_images_from_docling_doc(doc, pdf_path, image_output_root)

    chunks = _chunk_docling_document(doc, metadata['title'], images)
    print(f"  Created {len(chunks)} chunks")

    if not chunks:
        return {"metadata": metadata, "chunks": [], "embeddings": [], "images": images}

    texts_to_embed = [chunk["content"] for chunk in chunks]
    embeddings = get_embeddings(texts_to_embed, embedding_service_url)

    return {
        "metadata": metadata,
        "chunks": chunks,
        "embeddings": embeddings,
        "images": images,
    }


def list_pdfs(pdf_directory: str) -> List[Path]:
    """
    List all PDF files in a directory.

    Args:
        pdf_directory: Path to directory containing PDFs

    Returns:
        List of Path objects for each PDF file
    """
    return list(Path(pdf_directory).glob("*.pdf"))


# ============================================
# TESTING
# ============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PDF Processing utilities")
    parser.add_argument("--pdf", type=str, help="Path to a single PDF to process")
    parser.add_argument("--dir", type=str, help="Path to directory of PDFs to list")
    parser.add_argument("--no-images", action="store_true", help="Skip image extraction")
    parser.add_argument("--show-figures", action="store_true", help="Show chunks with figure references")
    args = parser.parse_args()

    if args.pdf:
        result = process_pdf(args.pdf, extract_images=not args.no_images)
        print(f"\nMetadata: {result['metadata']}")
        print(f"Chunks: {len(result['chunks'])}")
        print(f"Embeddings: {len(result['embeddings'])}")
        print(f"Images extracted: {len(result.get('images', []))}")

        if result['chunks']:
            chunk = result['chunks'][0]
            print(f"\nFirst chunk preview:")
            print(f"  Content: {chunk['content'][:200]}...")
            print(f"  Section: {chunk['section']}")
            print(f"  Page: {chunk.get('page', 'N/A')}")
            print(f"  has_table: {chunk['has_table']}, has_figure: {chunk['has_figure']}")
            print(f"  figure_ids: {chunk.get('figure_ids', [])}")
            print(f"  table_ids: {chunk.get('table_ids', [])}")
            print(f"  image_paths: {chunk.get('image_paths', [])}")

        if args.show_figures:
            print("\n--- Chunks with figure references ---")
            for i, chunk in enumerate(result['chunks']):
                if chunk.get('figure_ids') or chunk.get('image_paths'):
                    print(f"\nChunk {i}:")
                    print(f"  Section: {chunk['section']}")
                    print(f"  Page: {chunk.get('page', 'N/A')}")
                    print(f"  figure_ids: {chunk.get('figure_ids', [])}")
                    print(f"  image_paths: {chunk.get('image_paths', [])}")
                    print(f"  Preview: {chunk['original_content'][:150]}...")

        if result.get('images'):
            print("\n--- Extracted Images ---")
            for img in result['images'][:5]:
                print(f"  Page {img['page']}: {img['path']}")
                print(f"    figure_id: {img.get('figure_id')}, size: {img['width']}x{img['height']}")
            if len(result['images']) > 5:
                print(f"  ... and {len(result['images']) - 5} more images")

    if args.dir:
        pdfs = list_pdfs(args.dir)
        print(f"Found {len(pdfs)} PDFs:")
        for pdf in pdfs:
            print(f"  - {pdf.name}")
