"""
Protein Design Agent
~~~~~~~~~~~~~~~~~~~~
Agent logic, tools, and graph construction for enzyme analysis + literature RAG.
Supports both ChromaDB and Pinecone as RAG backends, plus MCP servers for arXiv,
bioRxiv, UniProt, EC, and PDB lookups.
"""

from typing import TypedDict, Annotated, Union
from functools import lru_cache
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import sys
import os

# Add project root to path to import mcp_servers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_servers.ec.server import ECMCPServer
from mcp_servers.pdb.server import PdbMCPServer
from mcp_servers.arxiv.server import ArxivMCPServer
from mcp_servers.biorxiv.server import BiorxivMCPServer
from mcp_servers.uniprot.server import UniProtMCPServer

# Import RAG backends
from agent.rag.chroma_rag import ChromaPDFRAG
from agent.rag.pinecone_rag import PineconePDFRAG

# Initialize MCP servers
ec_server = ECMCPServer()
pdb_server = PdbMCPServer()
arxiv_server = ArxivMCPServer()
biorxiv_server = BiorxivMCPServer()
uniprot_server = UniProtMCPServer()


import dotenv
dotenv.load_dotenv()

# ============================================
# TOOL DEFINITIONS
# ============================================


@lru_cache(maxsize=256)
def _cached_ec_lookup(enzyme_name: str):
    """Cached wrapper around EC lookup by name."""
    normalized = enzyme_name.strip()
    return ec_server.get_ec_number_by_name(normalized)


@lru_cache(maxsize=256)
def _cached_structure_lookup(enzyme_name: str):
    """Cached wrapper around PDB structure search."""
    normalized = enzyme_name.strip()
    return pdb_server.search_structure(normalized, max_results=1)

@tool
def get_ec_number(enzyme_name: str) -> str:
    """Get EC number for an enzyme. Use for simple lookups."""
    results = _cached_ec_lookup(enzyme_name)
    if results:
        # Return the best match (first result)
        best = results[0]
        enzyme_names = best.get('enzyme_names', [])
        name_str = enzyme_names[0] if enzyme_names else enzyme_name
        sources = ", ".join(best.get('sources', []))
        return f"EC {best['ec_number']} for {name_str} (Source: {sources})"
    return f"EC number not found for {enzyme_name}"


@tool
def get_enzyme_structure(enzyme_name: str) -> str:
    """Get 3D structure information for an enzyme."""
    # First search for PDB IDs associated with the enzyme name (cached)
    results = _cached_structure_lookup(enzyme_name)
    
    if results and "error" not in results[0]:
        best = results[0]
        pdb_id = best.get("pdb_id", "Unknown")
        title = best.get("title", "No title")
        resolution = best.get("resolution", "N/A")
        method = best.get("method", "Unknown")
        organism = ", ".join(best.get("organism", []))
        
        return f"PDB ID: {pdb_id}, Resolution: {resolution} Å, Organism: {organism}, Method: {method}, Title: {title}"
    
    return f"Structure not found for {enzyme_name}"


# @tool
# def get_catalytic_mechanism(ec_number: str) -> str:
#     """Get detailed catalytic mechanism for an EC number."""
#     # Clean EC number
#     ec_clean = ec_number.replace("EC", "").strip()
    
#     info = ec_server.lookup_ec_number(ec_clean)
    
#     if "error" not in info:
#         reaction = info.get("reaction", "No reaction data available.")
#         comments = "\n".join(info.get("comments", []))
#         return f"Reaction: {reaction}\nDetails: {comments}"
        
#     return f"Mechanism not available for {ec_number}: {info.get('error')}"


# ============================================
# NEW MCP SERVER TOOLS (arXiv, bioRxiv, UniProt)
# ============================================

@tool
def search_arxiv_papers(query: str, max_results: int = 5) -> str:
    """
    Search arXiv for scientific papers on protein/enzyme engineering, machine learning
    for biology, or related topics. Returns paper titles, abstracts, authors, and URLs.
    Use for finding recent research publications and methods.
    """
    results = arxiv_server.search_papers(query, max_results=max_results)
    
    if isinstance(results, dict) and "error" in results:
        return f"arXiv search failed: {results['error']}"
    
    if not results:
        return "No arXiv papers found for this query."
    
    lines = []
    for i, paper in enumerate(results, 1):
        title = paper.get("title", "Unknown")
        authors = ", ".join(paper.get("authors", [])[:3])
        if len(paper.get("authors", [])) > 3:
            authors += " et al."
        published = paper.get("published", "")
        abstract = paper.get("abstract", "")[:300] + "..." if len(paper.get("abstract", "")) > 300 else paper.get("abstract", "")
        url = paper.get("url", "")
        
        lines.append(
            f"[{i}] {title}\n"
            f"    Authors: {authors}\n"
            f"    Published: {published}\n"
            f"    URL: {url}\n"
            f"    Abstract: {abstract}\n"
        )
    
    return "\n".join(lines).strip()


@tool
def search_preprints(query: str, max_results: int = 5, source: str = "all") -> str:
    """
    Search bioRxiv/medRxiv and related databases for preprints on enzyme/protein design.
    Also searches RCSB PDB for structure-related papers.
    
    Args:
        query: Search keywords (e.g., 'chalcone isomerase', 'de novo design')
        max_results: Maximum number of results
        source: 'all', 'europepmc', 'rcsb', or 'biorxiv'
    """
    results = biorxiv_server.integrated_search(query, max_results=max_results, source=source)
    
    if not results:
        return "No preprints found for this query."
    
    lines = []
    for i, paper in enumerate(results, 1):
        title = paper.get("title", "Unknown")
        authors = paper.get("authors", "Unknown")
        doi = paper.get("doi", "")
        pub_source = paper.get("source", "")
        method = paper.get("search_method", "")
        pdb_id = paper.get("related_pdb", "")
        
        line = f"[{i}] {title}\n    Authors: {authors}\n    DOI: {doi}\n    Source: {pub_source}"
        if pdb_id:
            line += f"\n    Related PDB: {pdb_id}"
        lines.append(line + "\n")
    
    return "\n".join(lines).strip()


@tool
def search_uniprot_proteins(query: str, max_results: int = 5, organism: str = None) -> str:
    """
    Search UniProt for proteins by name, function, or keywords. Returns protein sequences,
    organism, EC numbers, and functional annotations.
    
    Args:
        query: Search query (e.g., 'chalcone isomerase', 'thermostable lipase')
        max_results: Maximum number of results
        organism: Optional filter by organism (e.g., 'Homo sapiens', 'Escherichia coli')
    """
    results = uniprot_server.search_proteins(query, max_results=max_results, organism=organism)
    
    if isinstance(results, dict) and "error" in results:
        return f"UniProt search failed: {results['error']}"
    
    if not results:
        return "No UniProt proteins found for this query."
    
    lines = []
    for i, protein in enumerate(results, 1):
        name = protein.get("protein_name", "Unknown")
        uniprot_id = protein.get("uniprot_id", "")
        org = protein.get("organism", "Unknown")
        ec_nums = ", ".join(protein.get("ec_numbers", []))
        length = protein.get("length", 0)
        function = protein.get("function", "")[:200] + "..." if len(protein.get("function", "")) > 200 else protein.get("function", "")
        sequence = protein.get("sequence", "")[:60] + "..." if len(protein.get("sequence", "")) > 60 else protein.get("sequence", "")
        
        line = f"[{i}] {name}\n    UniProt ID: {uniprot_id}\n    Organism: {org}\n    Length: {length} aa"
        if ec_nums:
            line += f"\n    EC: {ec_nums}"
        if function:
            line += f"\n    Function: {function}"
        line += f"\n    Sequence: {sequence}\n"
        lines.append(line)
    
    return "\n".join(lines).strip()


@tool
def get_uniprot_protein_details(uniprot_id: str) -> str:
    """
    Get detailed information about a specific protein by UniProt ID, including
    sequence, active sites, binding sites, and PDB structures.
    """
    result = uniprot_server.get_protein_by_id(uniprot_id)
    
    if isinstance(result, dict) and "error" in result:
        return f"Failed to get protein details: {result['error']}"
    
    lines = [
        f"Protein: {result.get('protein_name', 'Unknown')}",
        f"UniProt ID: {result.get('uniprot_id', '')}",
        f"Organism: {result.get('organism', 'Unknown')}",
        f"Length: {result.get('length', 0)} aa",
        f"Mass: {result.get('mass', 'N/A')} Da",
    ]
    
    ec_nums = result.get("ec_numbers", [])
    if ec_nums:
        lines.append(f"EC Numbers: {', '.join(ec_nums)}")
    
    function = result.get("function", "")
    if function:
        lines.append(f"Function: {function}")
    
    active_sites = result.get("active_sites", [])
    if active_sites:
        sites_str = "; ".join([f"Pos {s['position']}: {s['description']}" for s in active_sites[:5]])
        lines.append(f"Active Sites: {sites_str}")
    
    pdb_ids = result.get("pdb_structures", [])
    if pdb_ids:
        lines.append(f"PDB Structures: {', '.join(pdb_ids)}")
    
    sequence = result.get("sequence", "")
    if sequence:
        lines.append(f"Sequence: {sequence[:100]}{'...' if len(sequence) > 100 else ''}")
    
    return "\n".join(lines)


# ============================================
# RAG BACKEND FACTORY
# ============================================

def create_rag_backend() -> Union[ChromaPDFRAG, PineconePDFRAG]:
    """
    Create the appropriate RAG backend based on configuration.
    
    Uses RAG_BACKEND environment variable to choose between 'chroma' (default)
    and 'pinecone'.
    """
    backend = os.getenv("RAG_BACKEND", "chroma").lower()
    
    if backend == "pinecone":
        print("Using Pinecone RAG backend")
        return PineconePDFRAG(
            pdf_directory=os.getenv("RAG_PDF_DIR", "./data/papers"),
            bm25_persist_directory=os.getenv("RAG_BM25_DIR", "./data/bm25_pinecone"),
            embedding_service_url=os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8011/embed"),
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "paper-rag-index"),
            pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "papers"),
            pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
            pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
            use_hybrid_search=os.getenv("RAG_HYBRID_SEARCH", "true").lower() == "true",
        )
    else:
        print("Using Chroma RAG backend")
        return ChromaPDFRAG(
            pdf_directory=os.getenv("RAG_PDF_DIR", "./data/papers"),
            vd_persist_directory=os.getenv("RAG_CHROMA_DIR", "./data/chroma_db"),
            bm25_persist_directory=os.getenv("RAG_BM25_DIR", "./data/bm25"),
            embedding_service_url=os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8011/embed"),
        )


# ============================================
# STATE DEFINITION
# ============================================

class AgentState(TypedDict):
    """State for the protein design agent."""
    messages: Annotated[list, add_messages]
    query_type: str  # "simple" | "detailed" | "research"
    skill_name: str  # key into SKILL_REGISTRY, or "" for no skill
    needs_rag: bool
    rag_context: str  # formatted context from retrieved papers (if any)


# ============================================
# SKILL REGISTRY
# ============================================

def load_skill_content(file_path: str) -> str:
    """Load skill content from a markdown file."""
    try:
        # Resolve path relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Check if path is already absolute
        if os.path.isabs(file_path):
            abs_path = file_path
        else:
            abs_path = os.path.join(current_dir, file_path)
            
        with open(abs_path, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Failed to load skill file {file_path}: {e}")
        return ""


def _build_skill_registry() -> dict[str, dict]:
    """
    Auto-discover all .md files in the skills/ directory and build a registry.
    
    Each skill gets:
      - name: derived from filename (e.g. "EnzymeAnalysis.md" -> "enzyme_analysis")
      - content: the full markdown text
      - description: first non-empty, non-heading line from the file (used by the router)
    
    Returns:
        Dict mapping skill_name -> {"content": str, "description": str, "file": str}
    """
    skills_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")
    registry: dict[str, dict] = {}
    
    if not os.path.isdir(skills_dir):
        print(f"Warning: skills directory not found at {skills_dir}")
        return registry
    
    for filename in sorted(os.listdir(skills_dir)):
        if not filename.endswith(".md"):
            continue
        
        # Derive a snake_case key from the filename: "EnzymeAnalysis.md" -> "enzyme_analysis"
        raw_name = filename.removesuffix(".md")
        # Insert underscore before uppercase letters, then lowercase
        import re as _re
        skill_name = _re.sub(r'(?<=[a-z0-9])([A-Z])', r'_\1', raw_name).lower()
        
        content = load_skill_content(os.path.join(skills_dir, filename))
        if not content:
            continue
        
        # Extract a short description: first non-blank, non-heading line
        description = ""
        for line in content.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                description = stripped[:120]
                break
        if not description:
            description = f"Skill loaded from {filename}"
        
        registry[skill_name] = {
            "content": content,
            "description": description,
            "file": filename,
        }
        print(f"  Registered skill: {skill_name} ({filename})")
    
    return registry


# Build the registry once at import time
SKILL_REGISTRY = _build_skill_registry()

# ============================================
# AGENT GRAPH BUILDER
# ============================================

def build_agent(model: str = "gpt-4o-mini", temperature: float = 0):
    """
    Build the protein design agent graph.
    
    Args:
        model: OpenAI model name
        temperature: LLM temperature
        
    Returns:
        Compiled LangGraph agent
    """
    
    api_key = os.getenv("ZENMUX_API_KEY")
    base_url = os.getenv("ZENMUX_BASE_URL", "https://api.zenmux.com/v1")

    if not api_key:
        raise ValueError("ZENMUX_API_KEY not found in environment variables")

    # Split models: small router + stronger worker (both via Zenmux OpenAI-compatible endpoint)
    # Defaults: keep backwards-compat by falling back to LLM_MODEL or the function arg.
    router_model = os.getenv("ROUTER_MODEL", os.getenv("LLM_MODEL", model))
    worker_model = os.getenv("WORKER_MODEL", os.getenv("LLM_MODEL", model))

    router_llm = ChatOpenAI(
        model=router_model,
        temperature=0,  # routing should be deterministic
        openai_api_key=api_key,
        openai_api_base=base_url,
    )
    worker_llm = ChatOpenAI(
        model=worker_model,
        temperature=1,
        openai_api_key=api_key,
        openai_api_base=base_url,
    )

    # #region agent log
    import json as _json
    with open("/home/kellywang/project/protein_projects/protein-design-agent/.cursor/debug.log", "a") as _f:
        _f.write(_json.dumps({"hypothesisId": "H-A,H-B,H-D", "location": "agent.py:build_agent", "message": "LLM config", "data": {"router_model": router_model, "worker_model": worker_model, "router_temp": 0, "worker_temp": temperature}, "sessionId": "debug-session", "runId": "run1"}) + "\n")
    # #endregion

    # Initialize RAG backend (Chroma or Pinecone based on RAG_BACKEND env var)
    rag = create_rag_backend()
    rag_backend_type = os.getenv("RAG_BACKEND", "chroma").lower()
    print(f"Using RAG backend: {rag_backend_type}")

    @tool
    def search_research_papers(query: str, top_k: int = 5) -> str:
        """
        Search local indexed research papers for relevant information (model architectures,
        protein design methods, structure improvement strategies, etc). Use for complex
        research questions where tools alone are insufficient.
        """
        results = rag.search(query, top_k=top_k)
        if not results:
            return "No relevant local papers found."

        lines: list[str] = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "") or "Unknown title"
            section = r.get("section", "") or "Unknown section"
            filename = r.get("filename", "") or "Unknown file"
            score = r.get("score", None)
            content = r.get("content", "") or ""
            if len(content) > 1500:
                content = content[:1500] + "..."
            score_str = f"{score:.4f}" if isinstance(score, (float, int)) else "N/A"
            lines.append(
                f"[{i}] Title: {title}\n"
                f"    Section: {section}\n"
                f"    File: {filename}\n"
                f"    Score: {score_str}\n"
                f"    Excerpt:\n{content}\n"
            )

        return "\n".join(lines).strip()

    tools = [
        # Core enzyme tools
        get_ec_number, 
        get_enzyme_structure, 
        # Local RAG tool
        search_research_papers,
        # Literature search tools (MCP servers)
        search_arxiv_papers,
        search_preprints,
        # Protein sequence/annotation tools (MCP servers)
        search_uniprot_proteins,
        get_uniprot_protein_details,
    ]
    
    # ============ NODE DEFINITIONS ============
    
    def route_query(state: AgentState):
        """Classify query as simple, detailed, or research (needs RAG).
        
        Also selects a skill when the query is "detailed", so only the
        relevant skill instructions are injected into the prompt.
        """
        
        last_message = state["messages"][-1].content
        
        # ------------------------------------------------------------------
        # Build a dynamic catalogue of registered skills for the router
        # ------------------------------------------------------------------
        skill_lines: list[str] = []
        for sname, smeta in SKILL_REGISTRY.items():
            skill_lines.append(f'  - "{sname}": {smeta["description"]}')
        skill_catalogue = "\n".join(skill_lines) if skill_lines else "  (no skills registered)"
        
        prompt = f"""You are a query classifier for a protein engineering assistant.

User query: "{last_message}"

### Step 1 – Classify the query type
- "simple": user ONLY wants an EC number lookup (e.g. "EC number for X", "what's the EC of X")
- "detailed": user wants enzyme/protein details that can be answered via available tools (EC/PDB/mechanism/UniProt/arXiv/bioRxiv search) plus reasoning
- "research": user is asking a broader research question, machine learning model question, or methods question that should use local indexed papers via RAG first

Examples of "simple":
- "EC number for chalcone isomerase"
- "What's the EC of lipase?"

Examples of "detailed":
- "Tell me about lactate dehydrogenase"
- "Search arXiv for RFDiffusion papers"
- "Find UniProt entries for thermostable lipases"
- "Get PDB structure for hexokinase"

Examples of "research":
- "What's the model architecture of RFDiffusion?"
- "How can we improve the backbone structure of a protein?"
- "Summarize recent methods for improving enzyme thermostability"
- "Compare different protein design approaches from the literature"

### Step 2 – Select a skill (only when type is "detailed")
Available skills:
{skill_catalogue}

If the query clearly matches a skill, output that skill name.
If the query does NOT match any specific skill, output "none".
For "simple" or "research" queries, always output "none".

### Response format
Respond with EXACTLY two words separated by a space:
<query_type> <skill_name>

Examples:
  simple none
  detailed enzyme_analysis
  detailed none
  research none"""

        response = router_llm.invoke(prompt)
        raw = response.content.strip().lower().split()
        
        query_type = raw[0] if raw else "detailed"
        skill_name = raw[1] if len(raw) > 1 else "none"
        
        # #region agent log
        import json as _json
        with open("/home/kellywang/project/protein_projects/protein-design-agent/.cursor/debug.log", "a") as _f:
            _f.write(_json.dumps({"hypothesisId": "H-B", "location": "agent.py:route_query", "message": "router succeeded", "data": {"router_model": router_model, "query_type": query_type, "skill_name": skill_name}, "sessionId": "debug-session", "runId": "run1"}) + "\n")
        # #endregion
        
        # Validate query type
        if query_type not in ["simple", "detailed", "research"]:
            query_type = "detailed"  # safer default if unclear
        
        # Validate skill name – must exist in the registry or be "none"
        if skill_name not in SKILL_REGISTRY:
            skill_name = ""
        
        return {
            "query_type": query_type,
            "skill_name": skill_name,
            "needs_rag": query_type == "research",
        }
    
    
    def simple_handler(state: AgentState):
        """Handle simple EC number queries."""
        
        llm_with_tools = router_llm.bind_tools(tools)
        
        # Add instruction for simple queries if not already present
        messages = state["messages"]
        
        # Check if first message is a system message
        has_system = (len(messages) > 0 and 
                     hasattr(messages[0], 'type') and 
                     messages[0].type == "system")
        
        if not has_system:
            system_msg = SystemMessage(
                content=(
                    "User wants the EC number. Call get_ec_number() and respond with an informative sentence like: "
                    "'The EC number for {enzyme name} is EC X.X.X.X (Source: {source}).' "
                    "Include the full enzyme name and source from the tool result."
                )
            )
            messages = [system_msg] + messages
        
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    
    def detailed_handler(state: AgentState):
        """Handle detailed queries, injecting only the matched skill (if any)."""
        # #region agent log
        import json as _json
        with open("/home/kellywang/project/protein_projects/protein-design-agent/.cursor/debug.log", "a") as _f:
            _f.write(_json.dumps({"hypothesisId": "H-C", "location": "agent.py:detailed_handler", "message": "entering detailed_handler", "data": {"worker_model": worker_model, "worker_temp": temperature, "skill_name": state.get("skill_name", "")}, "sessionId": "debug-session", "runId": "run1"}) + "\n")
        # #endregion
        
        llm_with_tools = worker_llm.bind_tools(tools)
        
        # Add skill / RAG instructions if not already present
        messages = state["messages"]
        
        # Check if first message is a system message
        has_system = (len(messages) > 0 and 
                     hasattr(messages[0], 'type') and 
                     messages[0].type == "system")
        
        if not has_system:
            system_msgs: list[SystemMessage] = []
            
            # 1. Optional RAG context (from research pass)
            rag_context = (state.get("rag_context") or "").strip()
            if rag_context:
                system_msgs.append(
                    SystemMessage(
                        content=(
                            "Use the following retrieved paper excerpts as grounded context. "
                            "Cite them as [#] when relevant, and prefer them over speculation.\n\n"
                            f"{rag_context}"
                        )
                    )
                )
            
            # 2. Conditional skill injection — only if router selected one
            skill_name = (state.get("skill_name") or "").strip()
            if skill_name and skill_name in SKILL_REGISTRY:
                skill_content = SKILL_REGISTRY[skill_name]["content"]
                system_msgs.append(SystemMessage(content=skill_content))
            else:
                # Generic fallback: no special skill instructions, just be a
                # helpful protein engineering assistant.
                system_msgs.append(
                    SystemMessage(
                        content=(
                            "You are an expert protein engineering assistant. "
                            "Use the available tools to answer the user's question "
                            "thoroughly, and present your findings in well-structured markdown."
                        )
                    )
                )
            
            messages = system_msgs + messages
        
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def rag_handler(state: AgentState):
        """Retrieve relevant paper context for research-style questions."""
        query = state["messages"][-1].content
        rag_context = search_research_papers.invoke({"query": query, "top_k": 5})
        return {
            "rag_context": rag_context,
            "needs_rag": False,
        }
    
    
    # Tool execution node
    tool_node = ToolNode(tools)
    
    
    # ============ ROUTING FUNCTIONS ============
    
    def route_based_on_type(state: AgentState):
        """Route to simple or detailed handler based on query type."""
        if state["query_type"] == "simple":
            return "simple_handler"
        if state["query_type"] == "research":
            return "rag_handler"
        return "detailed_handler"
    
    
    def should_continue(state: AgentState):
        """Check if we need to call tools."""
        last_message = state["messages"][-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END
    
    
    def route_after_tools(state: AgentState):
        """After tools execute, route back to appropriate handler."""
        if state["query_type"] == "simple":
            return "simple_handler"
        return "detailed_handler"
    
    
    # ============ BUILD GRAPH ============
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("route_query", route_query)
    workflow.add_node("simple_handler", simple_handler)
    workflow.add_node("rag_handler", rag_handler)
    workflow.add_node("detailed_handler", detailed_handler)
    workflow.add_node("tools", tool_node)
    
    # Define edges
    workflow.add_edge(START, "route_query")
    
    # Route to appropriate handler after classification
    workflow.add_conditional_edges(
        "route_query",
        route_based_on_type,
        {
            "simple_handler": "simple_handler",
            "rag_handler": "rag_handler",
            "detailed_handler": "detailed_handler"
        }
    )

    # Research path: retrieve papers then proceed to detailed handler
    workflow.add_edge("rag_handler", "detailed_handler")
    
    # Both handlers may need to call tools
    workflow.add_conditional_edges("simple_handler", should_continue)
    workflow.add_conditional_edges("detailed_handler", should_continue)
    
    # After tools, route back to handler for response generation
    workflow.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "simple_handler": "simple_handler",
            "detailed_handler": "detailed_handler"
        }
    )
    
    # Use a checkpointer so the agent remembers prior turns per thread_id.
    # Each call with the same thread_id will load the previous messages
    # and append the new ones, giving the LLM full conversation context.
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ============================================
# UTILITY FUNCTIONS
# ============================================

def extract_final_response(final_state: AgentState) -> str | None:
    """
    Extract the final text response from agent state.
    
    Args:
        final_state: The final state from agent execution
        
    Returns:
        Final response text or None if not found
    """
    # Find last AI message that's not a tool call
    for msg in reversed(final_state["messages"]):
        if isinstance(msg, AIMessage):
            if msg.content and not msg.tool_calls:
                return msg.content
    
    return None


def run_agent_sync(agent, query: str) -> dict:
    """
    Run the agent synchronously and return results.
    
    Args:
        agent: Compiled LangGraph agent
        query: User query string
        
    Returns:
        Dictionary with response and metadata
    """
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "query_type": "",
        "needs_rag": False,
        "rag_context": "",
    }
    
    # Execute agent
    final_state = agent.invoke(initial_state)
    
    # Extract response
    response = extract_final_response(final_state)
    
    return {
        "response": response,
        "query_type": final_state.get("query_type"),
        "message_count": len(final_state["messages"]),
        "full_state": final_state
    }


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    # Build agent
    agent = build_agent()
    
    # Test simple query
    print("=" * 60)
    print("SIMPLE QUERY TEST")
    print("=" * 60)
    result = run_agent_sync(agent, "What's the EC number for chalcone isomerase?")
    print(f"Query Type: {result['query_type']}")
    print(f"Response: {result['response']}")
    print()
    
    # Test detailed query
    print("=" * 60)
    print("DETAILED QUERY TEST")
    print("=" * 60)
    result = run_agent_sync(agent, "Tell me about lactate dehydrogenase")
    print(f"Query Type: {result['query_type']}")
    print(f"Response: {result['response']}")