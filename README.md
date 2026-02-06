# 🧬 Protein Design Research Agent

An AI-powered assistant for protein engineering and enzyme design research. This tool integrates **RAG (Retrieval-Augmented Generation)** with **Agentic Workflows** (powered by LangGraph & Zenmux/Gemini) to help researchers find papers, query protein databases, and synthesize information.

## 🚀 Key Features

### � Enhanced RAG System
The system uses an advanced Retrieval-Augmented Generation pipeline (`agent/rag/enhanced_paper_rag.py`) designed for scientific literature:
*   **Hybrid Search**: Combines **Vector Search** (semantic similarity via `pubmedbert-base-embeddings`) and **Keyword Search** (BM25) using Reciprocal Rank Fusion (RRF) for high-precision retrieval.
*   **Markdown Parsing**: Uses `pymupdf4llm` to convert PDFs into structured Markdown, preserving headers, tables, and lists better than plain text extraction.
*   **Hierarchical Chunking**: Splits documents by headers first, then paragraphs, ensuring chunks respect document structure.
*   **Context Enrichment**: Every chunk is tagged with its source paper title and section hierarchy (e.g., `Methods > Protein Purification`) for better LLM context.

### 🤖 Agentic Workflow (LangGraph)
The agent (`agent/enzyme_agent.py`) follows a sophisticated reasoning loop:
1.  **Tools First Strategy**: It first attempts to answer queries using precise tools (e.g., "What is the EC number of X?") without hallucinating.
2.  **Fallback to RAG**: If tools are insufficient, it retrieves context from your local PDF library.
3.  **Online Search Fallback**: If local papers don't cover the topic, it automatically searches **bioRxiv/EuropePMC** for the latest preprints.

### � Integrated MCP Servers
The project adopts a modular **Model Context Protocol (MCP)** style architecture for tools:
*   **🧬 UniProt Server**:
    *   Search proteins by name, function, or gene.
    *   Retrieve full amino acid sequences, organism details, and lengths.
    *   *New*: Robust query handling and "Click-to-View" results in the UI.
*   **🔢 EC Server**:
    *   Look up Enzyme Commission numbers by name.
    *   Get reaction patterns and systematic names.
*   **� BioRxiv/EuropePMC Server**:
    *   Search for recent preprints and publications.
    *   Retrieve abstracts and DOIs.

## 🛠️ Technology Stack

*   **Frontend**: [Streamlit](https://streamlit.io/)
*   **LLM**: Zenmux (Gemini 2.5 Flash Lite) via [LangChain](https://www.langchain.com/)
*   **Orchestration**: [LangGraph](https://langchain-ai.github.io/langgraph/)
*   **Vector DB**: [ChromaDB](https://www.trychroma.com/)
*   **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
*   **PDF Processing**: PyMuPDF

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd protein-design-agent
    ```

2.  **Install dependencies** (using `uv` is recommended):
    ```bash
    uv sync
    # OR with pip
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables**:
    Create a `.env` file in the root directory:
    ```bash
    ZENMUX_API_KEY=your_zenmux_api_key_here
    # Optional: ZENMUX_BASE_URL if using a custom endpoint
    ```

## 🏃‍♂️ Usage

### 1. Index Your Papers
Place your PDF research papers in the `data/papers` directory. Then run the indexing script:

```bash
uv run agent/rag/enhanced_paper_rag.py --index
```
*This will chunk and embed the PDFs into `data/chroma_db`.*

### 2. Run the Web Interface
Start the Streamlit app:

```bash
streamlit run app.py
```
*Access the app at `http://localhost:8501`.*

### 3. Example Queries
*   **Database Lookup**: "What is the amino acid sequence of Hexokinase-1 from Homo sapiens?"
*   **EC Info**: "What is the EC number for chalcone isomerase?"
*   **Literature RAG**: "How can I improve the thermostability of hydrolases?"
*   **Online Search**: "Find recent papers about 'Evo 2' on bioRxiv."

## 📂 Project Structure

*   `app.py`: Main Streamlit application entry point.
*   `agent/`: Core agent logic.
    *   `enzyme_agent.py`: LangGraph agent definition and tool integration.
    *   `rag/`: RAG implementation (PDF loading, chunking, retrieval).
*   `mcp_servers/`: Modular tools (Model Context Protocol style).
    *   `ec/`: EC number lookup tools.
    *   `uniprot/`: UniProt API integration.
    *   `biorxiv/`: BioRxiv/EuropePMC search tools.
*   `data/`: Data storage.
    *   `papers/`: Directory for your PDF files.
    *   `chroma_db/`: Persisted vector database.

## 🤝 Contributing
Feel free to open issues or submit pull requests for new tools (e.g., BRENDA, PDB integration) or improvements to the agent's reasoning capabilities.
