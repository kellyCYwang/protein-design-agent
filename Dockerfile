# =============================================================================
# Protein Design Agent - Main Application
# =============================================================================
# Multi-stage build for smaller, more secure image

FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster dependency management
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml ./

# Create requirements.txt from pyproject.toml
RUN uv pip compile pyproject.toml -o requirements.txt --python-version 3.11

# Install dependencies into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Runtime stage
# =============================================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Copy application code
COPY --chown=appuser:appuser agent/ ./agent/
COPY --chown=appuser:appuser mcp_servers/ ./mcp_servers/
COPY --chown=appuser:appuser app.py ./
COPY --chown=appuser:appuser public/ ./public/

# Create data directories (will be mounted as volumes in K8s)
RUN mkdir -p /app/data/papers /app/data/chroma_db /app/data/bm25 /app/data/pdf_images \
    && chown -R appuser:appuser /app/data

# Switch to non-root user
USER appuser

# Environment variables with defaults
ENV RAG_PDF_DIR=/app/data/papers \
    RAG_CHROMA_DIR=/app/data/chroma_db \
    RAG_BM25_DIR=/app/data/bm25 \
    RAG_IMAGE_DIR=/app/data/pdf_images \
    RAG_BACKEND=chroma \
    EMBEDDING_SERVICE_URL=http://embedding-service:8000/embed \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
