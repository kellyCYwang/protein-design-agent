I will implement the embedding model as a separate microservice as requested.

**Revised Plan:**

1. **Create Embedding Microservice (`embedding_service.py`)**:

   * Use **FastAPI** and **Uvicorn** to create a lightweight server.

   * Load `SentenceTransformer('neuml/pubmedbert-base-embeddings')` **once** at startup.

   * Expose a `/embed` endpoint that accepts text (or list of texts) and returns vector embeddings.

   * This service will run independently (e.g., on port 8000).

2. **Update RAG System (`agent/rag/enhanced_paper_rag.py`)**:

   * Modify `LocalPDFRAG` to **remove** the direct `SentenceTransformer` dependency.

   * Replace `self.embedding_model.encode()` calls with HTTP requests to `http://localhost:8000/embed`.

   * Implement a check: if the service is unreachable, fail gracefully or (optionally) fall back to local loading (but we will prioritize the service pattern).

3. **Production Context Explanation**:

   * I will explain that in production, this service would be a Docker container (e.g., `embedding-service`) that scales independently from the main app.

**Files to Create/Modify:**

* `embedding_service.py` (New)

* `agent/rag/enhanced_paper_rag.py` (Modify)

**Execution Steps:**

1. Write `embedding_service.py`.
2. Start the service in a background terminal.
3. Modify `enhanced_paper_rag.py` to consume the API.
4. Verify the speedup in the main app (since it no longer loads the model).

