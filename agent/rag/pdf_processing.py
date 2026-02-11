#!/usr/bin/env python3
"""
PDF Processing Module
~~~~~~~~~~~~~~~~~~~~~
Shared utilities for PDF parsing, markdown conversion, chunking, embedding,
and image extraction with figure-chunk linking.
Used by both ChromaPDFRAG and PineconePDFRAG backends.
"""

import os
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import pymupdf4llm
import fitz  # PyMuPDF
import requests
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


# ============================================
# CONSTANTS
# ============================================

# Regex patterns for detecting figure/table references
FIGURE_PATTERN = re.compile(r"(?:Figure|Fig\.?)\s*(\d+[A-Za-z]?)", re.IGNORECASE)
TABLE_PATTERN = re.compile(r"Table\s*(\d+[A-Za-z]?)", re.IGNORECASE)

# Default image output directory (overridable via RAG_IMAGE_DIR env var)
DEFAULT_IMAGE_OUTPUT_ROOT = os.getenv("RAG_IMAGE_DIR", "./data/pdf_images")


# ============================================
# EMBEDDING SERVICE
# ============================================

# Default embedding service URL (overridable via EMBEDDING_SERVICE_URL env var)
DEFAULT_EMBEDDING_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8011/embed")


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
# PDF METADATA EXTRACTION
# ============================================

def extract_metadata_from_pdf(pdf_path: str) -> Dict:
    """
    Extract metadata from a PDF file using PyMuPDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary with title, author, and filename
    """
    try:
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        title = metadata.get('title', '') or metadata.get('Title', '')
        if not title:
            title = Path(pdf_path).stem.replace('_', ' ').replace('-', ' ')
        author = metadata.get('author', '') or metadata.get('Author', 'Unknown')
        doc.close()
        return {"title": title, "author": author, "filename": Path(pdf_path).name}
    except Exception:
        return {"title": Path(pdf_path).stem, "author": "Unknown", "filename": Path(pdf_path).name}


# ============================================
# IMAGE EXTRACTION
# ============================================

def _find_figure_caption_near_image(
    page: fitz.Page,
    image_bbox: fitz.Rect,
    search_distance: float = 50.0
) -> Optional[str]:
    """
    Search for figure caption text near an image bounding box.
    
    Args:
        page: PyMuPDF page object
        image_bbox: Bounding box of the image
        search_distance: How far below/above image to search for caption (points)
        
    Returns:
        Figure ID string (e.g. "Figure 3") if found, else None
    """
    # Get all text blocks on the page
    blocks = page.get_text("blocks")
    
    for block in blocks:
        # block format: (x0, y0, x1, y1, text, block_no, block_type)
        if len(block) < 5:
            continue
        
        bx0, by0, bx1, by1, text = block[:5]
        
        # Check if text block is near the image (below or above)
        # Below the image
        if (by0 >= image_bbox.y1 and 
            by0 <= image_bbox.y1 + search_distance and
            abs(bx0 - image_bbox.x0) < 100):  # roughly aligned horizontally
            
            match = FIGURE_PATTERN.search(text)
            if match:
                return f"Figure {match.group(1)}"
        
        # Above the image (some papers put captions above)
        if (by1 <= image_bbox.y0 and 
            by1 >= image_bbox.y0 - search_distance and
            abs(bx0 - image_bbox.x0) < 100):
            
            match = FIGURE_PATTERN.search(text)
            if match:
                return f"Figure {match.group(1)}"
    
    return None


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
        List of image records with keys:
        - page: Page number (0-indexed)
        - path: File path where image was saved
        - figure_id: Detected figure label (e.g. "Figure 3") or None
        - width: Image width in pixels
        - height: Image height in pixels
        - xref: PDF internal reference ID
    """
    paper_id = Path(pdf_path).stem
    paper_image_dir = Path(output_root) / paper_id
    paper_image_dir.mkdir(parents=True, exist_ok=True)
    
    images: List[Dict] = []
    
    try:
        doc = fitz.open(pdf_path)
        
        for page_index, page in enumerate(doc):
            # Get all images on this page
            image_list = page.get_images(full=True)
            
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                
                try:
                    # Extract the image
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image.get("ext", "png")
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)
                    
                    # Skip tiny images (likely icons or decorations)
                    if width < min_width or height < min_height:
                        continue
                    
                    # Save image to disk
                    image_filename = f"page_{page_index}_img_{img_index}.{image_ext}"
                    image_path = paper_image_dir / image_filename
                    
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                    
                    # Try to find figure caption
                    # Get image bounding box on page (approximate)
                    image_rects = page.get_image_rects(xref)
                    figure_id = None
                    if image_rects:
                        # Use first rect if multiple
                        image_bbox = image_rects[0]
                        figure_id = _find_figure_caption_near_image(page, image_bbox)
                    
                    images.append({
                        "page": page_index,
                        "path": str(image_path),
                        "figure_id": figure_id,
                        "width": width,
                        "height": height,
                        "xref": xref
                    })
                    
                except Exception as e:
                    print(f"    Warning: Could not extract image {img_index} on page {page_index}: {e}")
                    continue
        
        doc.close()
        
        print(f"  Extracted {len(images)} images to {paper_image_dir}")
        
    except Exception as e:
        print(f"  ❌ Image extraction failed: {e}")
    
    return images


# ============================================
# PDF TO MARKDOWN CONVERSION (with page_chunks for layout analysis)
# ============================================

def extract_text_as_markdown(pdf_path: str) -> str:
    """
    Convert PDF to Markdown using PyMuPDF4LLM with enhanced table/image handling.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Markdown text extracted from PDF
    """
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


def extract_pdf_with_page_chunks(
    pdf_path: str,
    image_output_dir: Optional[str] = None
) -> List[Dict]:
    """
    Extract PDF content using pymupdf4llm's page_chunks mode for improved layout analysis.
    
    This mode provides:
    - Per-page markdown with page numbers
    - Better multi-column handling
    - Automatic image extraction with proper naming
    - Table of contents awareness
    
    Args:
        pdf_path: Path to the PDF file
        image_output_dir: Directory for extracted images (if None, images not written)
        
    Returns:
        List of page chunk dictionaries, each containing:
        - metadata: dict with page number, images list, table of contents, etc.
        - text: markdown text for this page
        - tables: list of table objects on this page
        - images: list of image metadata on this page
    """
    try:
        print(f"  Parsing PDF with page_chunks: {Path(pdf_path).name}")
        
        # Configure image writing if output dir provided
        kwargs = {
            "page_chunks": True,  # Returns list of dicts, one per page
            "table_strategy": "lines_strict",  # Best for scientific tables
            "ignore_graphics": False,
            "show_progress": False,
        }
        
        if image_output_dir:
            Path(image_output_dir).mkdir(parents=True, exist_ok=True)
            kwargs["write_images"] = True
            kwargs["image_path"] = image_output_dir
        
        page_chunks = pymupdf4llm.to_markdown(pdf_path, **kwargs)
        
        print(f"  Extracted {len(page_chunks)} page chunks")
        return page_chunks
        
    except Exception as e:
        print(f"  ❌ Page chunk extraction failed: {e}")
        return []


def chunk_text_with_pages(
    page_chunks: List[Dict],
    title: str
) -> List[Dict]:
    """
    Hierarchical chunking from pymupdf4llm page_chunks output.
    
    This version preserves page numbers for better image-chunk linking.
    
    Args:
        page_chunks: Output from extract_pdf_with_page_chunks()
        title: Paper title for context enrichment
        
    Returns:
        List of chunk dictionaries with page numbers included
    """
    if not page_chunks:
        return []

    # Header splitter
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    # Character splitter for large sections
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    final_chunks = []
    
    for page_data in page_chunks:
        # Get page number (0-indexed in pymupdf4llm)
        page_metadata = page_data.get("metadata", {})
        page_num = page_metadata.get("page", 0)
        
        # Get page text
        page_text = page_data.get("text", "")
        if not page_text:
            continue
        
        # Get images on this page (from pymupdf4llm metadata)
        page_images = page_data.get("images", [])
        
        # Split by headers
        try:
            md_header_splits = markdown_splitter.split_text(page_text)
        except Exception:
            # Fallback: treat entire page as one section
            from langchain_core.documents import Document
            md_header_splits = [Document(page_content=page_text, metadata={})]
        
        for split in md_header_splits:
            # Build section context
            header_context = []
            if hasattr(split, 'metadata'):
                if "Header 1" in split.metadata:
                    header_context.append(split.metadata["Header 1"])
                if "Header 2" in split.metadata:
                    header_context.append(split.metadata["Header 2"])
                if "Header 3" in split.metadata:
                    header_context.append(split.metadata["Header 3"])
            
            section_str = " > ".join(header_context) if header_context else "General"
            
            # Get content
            content = split.page_content if hasattr(split, 'page_content') else str(split)
            
            # Sub-split large sections
            sub_splits = text_splitter.split_text(content)
            
            for sub_split in sub_splits:
                # Enrichment with page info
                enriched_content = f"Paper Title: {title} | Section: {section_str} | Page: {page_num + 1}\n\n{sub_split}"
                
                # Extract figure and table references
                figure_ids = _extract_figure_ids(sub_split)
                table_ids = _extract_table_ids(sub_split)
                
                # Detect content types
                has_markdown_table = _has_markdown_table_syntax(sub_split)
                has_image_md = _has_image_markdown(sub_split)
                
                has_table = bool(table_ids) or has_markdown_table
                has_figure = bool(figure_ids) or has_image_md
                
                final_chunks.append({
                    "content": enriched_content,
                    "original_content": sub_split,
                    "section": section_str,
                    "page": page_num,  # 0-indexed page number
                    "metadata": split.metadata if hasattr(split, 'metadata') else {},
                    "has_table": has_table,
                    "has_figure": has_figure,
                    "figure_ids": figure_ids,
                    "table_ids": table_ids,
                    "page_images": page_images,  # Images from this page
                })
    
    return final_chunks


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
            # Check for separator row like |---|---|
            if re.match(r"^\|?\s*[-:]+\s*\|", stripped):
                has_separator = True
    
    return has_pipe_row and has_separator


def _has_image_markdown(text: str) -> bool:
    """Check if text contains markdown image syntax ![...](...) or similar."""
    return bool(re.search(r"!\[.*?\]\(.*?\)", text))


def _extract_figure_ids(text: str) -> List[str]:
    """
    Extract all figure references from text.
    
    Returns normalized figure IDs like ["Figure 1", "Figure 2A", "Figure 3"]
    """
    matches = FIGURE_PATTERN.findall(text)
    # Normalize to "Figure X" format
    return [f"Figure {m}" for m in matches]


def _extract_table_ids(text: str) -> List[str]:
    """
    Extract all table references from text.
    
    Returns normalized table IDs like ["Table 1", "Table 2"]
    """
    matches = TABLE_PATTERN.findall(text)
    return [f"Table {m}" for m in matches]


# ============================================
# TEXT CHUNKING
# ============================================

def chunk_text(text: str, title: str) -> List[Dict]:
    """
    Hierarchical Chunking:
    1. Split by Headers (#, ##, ###)
    2. Split by Paragraphs (RecursiveCharacterTextSplitter)
    3. Enrich with Title + Section context
    4. Detect figure/table references and extract IDs
    
    Args:
        text: Markdown text to chunk
        title: Paper title for context enrichment
        
    Returns:
        List of chunk dictionaries with content, metadata, flags, and reference IDs:
        - content: Enriched text with title/section prefix
        - original_content: Raw chunk text
        - section: Section hierarchy string
        - metadata: Header metadata from splitter
        - has_table: True if chunk contains table content/references
        - has_figure: True if chunk contains figure content/references
        - figure_ids: List of figure IDs referenced (e.g. ["Figure 1", "Figure 2"])
        - table_ids: List of table IDs referenced (e.g. ["Table 1"])
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
            
            # Extract figure and table references using regex
            figure_ids = _extract_figure_ids(sub_split)
            table_ids = _extract_table_ids(sub_split)
            
            # Improved detection: check for actual content OR references
            has_markdown_table = _has_markdown_table_syntax(sub_split)
            has_image_md = _has_image_markdown(sub_split)
            
            has_table = bool(table_ids) or has_markdown_table
            has_figure = bool(figure_ids) or has_image_md
            
            final_chunks.append({
                "content": enriched_content,
                "original_content": sub_split,
                "section": section_str,
                "metadata": split.metadata,
                "has_table": has_table,
                "has_figure": has_figure,
                "figure_ids": figure_ids,  # e.g. ["Figure 1", "Figure 2A"]
                "table_ids": table_ids,    # e.g. ["Table 1"]
            })
    
    return final_chunks


# ============================================
# IMAGE-CHUNK LINKING
# ============================================

def _link_images_to_chunks(
    chunks: List[Dict],
    images: List[Dict],
    use_page_fallback: bool = True
) -> List[Dict]:
    """
    Link extracted images to chunks based on figure IDs and page proximity.
    
    Args:
        chunks: List of chunk dictionaries with figure_ids and optionally 'page'
        images: List of image records with figure_id and page
        use_page_fallback: If True, link images to chunks on same page when figure_id doesn't match
        
    Returns:
        Updated chunks with 'image_paths' field added
    """
    # Build figure_id -> image paths mapping
    figure_to_images: Dict[str, List[str]] = {}
    for img in images:
        fid = img.get("figure_id")
        if fid:
            figure_to_images.setdefault(fid, []).append(img["path"])
    
    # Build page -> image paths mapping (for page-based fallback)
    page_to_images: Dict[int, List[str]] = {}
    for img in images:
        page_to_images.setdefault(img["page"], []).append(img["path"])
    
    # Link images to chunks
    for chunk in chunks:
        image_paths: List[str] = []
        
        # Primary: match by figure IDs referenced in the chunk
        for fid in chunk.get("figure_ids", []):
            # Normalize figure ID for matching (handle "Figure 1" vs "Figure 1A" etc.)
            if fid in figure_to_images:
                image_paths.extend(figure_to_images[fid])
            else:
                # Try partial match (e.g. chunk says "Figure 1" but image is "Figure 1A")
                for img_fid, paths in figure_to_images.items():
                    if fid in img_fid or img_fid in fid:
                        image_paths.extend(paths)
        
        # Page-based fallback: if chunk has a page number and has_figure but no matched images
        if use_page_fallback and not image_paths:
            chunk_page = chunk.get("page")
            if chunk_page is not None and chunk.get("has_figure"):
                # Link all images from the same page
                if chunk_page in page_to_images:
                    image_paths.extend(page_to_images[chunk_page])
        
        # Deduplicate and sort
        chunk["image_paths"] = sorted(set(image_paths))
    
    return chunks


# ============================================
# BATCH PDF PROCESSING
# ============================================

def process_pdf(
    pdf_path: str,
    embedding_service_url: str = None,
    image_output_root: str = None,
    extract_images: bool = True,
    use_page_chunks: bool = True
) -> Dict:
    """
    Process a single PDF: extract metadata, convert to markdown, chunk, embed,
    and optionally extract images with figure-chunk linking.
    
    Args:
        pdf_path: Path to the PDF file
        embedding_service_url: URL of the embedding service
        image_output_root: Root directory for extracted images
        extract_images: Whether to extract images from PDF
        use_page_chunks: If True, use pymupdf4llm's page_chunks mode for better
                         layout analysis and page-based image linking
        
    Returns:
        Dictionary with:
        - metadata: PDF metadata (title, author, filename)
        - chunks: List of chunks with figure_ids, table_ids, image_paths, and page numbers
        - embeddings: List of embedding vectors for each chunk
        - images: List of extracted image records (if extract_images=True)
    """
    # Apply defaults from environment variables
    if embedding_service_url is None:
        embedding_service_url = DEFAULT_EMBEDDING_URL
    if image_output_root is None:
        image_output_root = DEFAULT_IMAGE_OUTPUT_ROOT
    
    print(f"\nProcessing: {Path(pdf_path).name}")
    
    # Extract metadata
    metadata = extract_metadata_from_pdf(pdf_path)
    print(f"  Title: {metadata['title'][:60]}...")
    
    paper_id = Path(pdf_path).stem
    paper_image_dir = str(Path(image_output_root) / paper_id) if extract_images else None
    
    # Extract images (if enabled)
    images: List[Dict] = []
    if extract_images:
        images = extract_images_from_pdf(pdf_path, output_root=image_output_root)
    
    chunks: List[Dict] = []
    
    if use_page_chunks:
        # Use page_chunks mode for improved layout analysis
        page_chunks = extract_pdf_with_page_chunks(
            pdf_path,
            image_output_dir=paper_image_dir if extract_images else None
        )
        
        if page_chunks:
            # Chunk with page information preserved
            chunks = chunk_text_with_pages(page_chunks, metadata['title'])
            print(f"  Created {len(chunks)} chunks (page_chunks mode)")
        else:
            # Fallback to standard extraction
            print("  Falling back to standard extraction...")
            use_page_chunks = False
    
    if not use_page_chunks or not chunks:
        # Standard extraction (fallback)
        md_text = extract_text_as_markdown(pdf_path)
        if not md_text:
            return {"metadata": metadata, "chunks": [], "embeddings": [], "images": images}
        
        chunks = chunk_text(md_text, metadata['title'])
        print(f"  Created {len(chunks)} chunks (standard mode)")
    
    if not chunks:
        return {"metadata": metadata, "chunks": [], "embeddings": [], "images": images}
    
    # Link images to chunks based on figure IDs and page proximity
    if images:
        # Page fallback only works if we have page info (page_chunks mode)
        use_page_fallback = use_page_chunks and any(c.get("page") is not None for c in chunks)
        chunks = _link_images_to_chunks(chunks, images, use_page_fallback=use_page_fallback)
        linked_count = sum(1 for c in chunks if c.get("image_paths"))
        print(f"  Linked images to {linked_count} chunks (page_fallback={use_page_fallback})")
    
    # Generate embeddings
    texts_to_embed = [chunk["content"] for chunk in chunks]
    embeddings = get_embeddings(texts_to_embed, embedding_service_url)
    
    return {
        "metadata": metadata,
        "chunks": chunks,
        "embeddings": embeddings,
        "images": images
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
    parser.add_argument("--no-page-chunks", action="store_true", 
                        help="Disable page_chunks mode (use standard extraction)")
    parser.add_argument("--show-figures", action="store_true", help="Show chunks with figure references")
    parser.add_argument("--show-pages", action="store_true", help="Show page distribution of chunks")
    args = parser.parse_args()
    
    if args.pdf:
        result = process_pdf(
            args.pdf, 
            extract_images=not args.no_images,
            use_page_chunks=not args.no_page_chunks
        )
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
        
        if args.show_pages:
            print("\n--- Page Distribution ---")
            page_counts: Dict[int, int] = {}
            for chunk in result['chunks']:
                page = chunk.get('page', -1)
                page_counts[page] = page_counts.get(page, 0) + 1
            for page in sorted(page_counts.keys()):
                page_label = f"Page {page + 1}" if page >= 0 else "Unknown"
                print(f"  {page_label}: {page_counts[page]} chunks")
        
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
            for img in result['images'][:5]:  # Show first 5
                print(f"  Page {img['page']}: {img['path']}")
                print(f"    figure_id: {img.get('figure_id')}, size: {img['width']}x{img['height']}")
            if len(result['images']) > 5:
                print(f"  ... and {len(result['images']) - 5} more images")
    
    if args.dir:
        pdfs = list_pdfs(args.dir)
        print(f"Found {len(pdfs)} PDFs:")
        for pdf in pdfs:
            print(f"  - {pdf.name}")
