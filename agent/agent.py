"""
Protein Design Agent
~~~~~~~~~~~~~~~~~~~~
Agent logic, tools, and graph construction for enzyme analysis + literature RAG.
"""

from typing import TypedDict, Annotated
from functools import lru_cache
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import sys
import os

# Add project root to path to import mcp_servers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_servers.ec.server import ECMCPServer
from mcp_servers.pdb.server import PdbMCPServer
from agent.rag.paper_rag import LocalPDFRAG

# Initialize servers
ec_server = ECMCPServer()
pdb_server = PdbMCPServer()


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


@tool
def get_catalytic_mechanism(ec_number: str) -> str:
    """Get detailed catalytic mechanism for an EC number."""
    # Clean EC number
    ec_clean = ec_number.replace("EC", "").strip()
    
    info = ec_server.lookup_ec_number(ec_clean)
    
    if "error" not in info:
        reaction = info.get("reaction", "No reaction data available.")
        comments = "\n".join(info.get("comments", []))
        return f"Reaction: {reaction}\nDetails: {comments}"
        
    return f"Mechanism not available for {ec_number}: {info.get('error')}"



# ============================================
# STATE DEFINITION
# ============================================

class AgentState(TypedDict):
    """State for the protein design agent."""
    messages: Annotated[list, add_messages]
    query_type: str  # "simple" | "detailed" | "research"
    needs_rag: bool
    rag_context: str  # formatted context from retrieved papers (if any)


# ============================================
# SKILL CONTENT
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
        # Return fallback content if file load fails
        return """You are an expert enzyme analyst. Provide comprehensive analysis.
        1. Get EC number
        2. Get Structure
        3. Get Mechanism
        """

# Load skill from file
DETAILED_ANALYSIS_SKILL = load_skill_content("skills/EnzymeAnalysis.md")

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

    # Initialize local paper RAG (hybrid search over PDFs already indexed)
    rag = LocalPDFRAG(
        pdf_directory=os.getenv("RAG_PDF_DIR", "./data/papers"),
        vd_persist_directory=os.getenv("RAG_CHROMA_DIR", "./data/chroma_db"),
        bm25_persist_directory=os.getenv("RAG_BM25_DIR", "./data/bm25"),
        embedding_service_url=os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8011/embed"),
    )

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

    tools = [get_ec_number, get_enzyme_structure, get_catalytic_mechanism, search_research_papers]
    
    # ============ NODE DEFINITIONS ============
    
    def route_query(state: AgentState):
        """Classify query as simple, detailed, or research (needs RAG)."""
        
        last_message = state["messages"][-1].content
        
        prompt = f"""You are a query classifier for a protein engineering assistant.

User query: "{last_message}"

Classify this as:
- "simple": user ONLY wants an EC number lookup (e.g. "EC number for X", "what's the EC of X")
- "detailed": user wants enzyme/protein details that can be answered via available tools (EC/PDB/mechanism) plus reasoning
- "research": user is asking a broader research question, model architecture question, or methods question that should use local papers via RAG first

Examples of "research":
- "What's the model architecture of RFDiffusion?"
- "How can we improve the backbone structure of a protein?"
- "Summarize recent methods for improving enzyme thermostability"

Respond with ONLY one word: simple, detailed, or research."""

        response = router_llm.invoke(prompt)
        query_type = response.content.strip().lower()
        # #region agent log
        import json as _json
        with open("/home/kellywang/project/protein_projects/protein-design-agent/.cursor/debug.log", "a") as _f:
            _f.write(_json.dumps({"hypothesisId": "H-B", "location": "agent.py:route_query", "message": "router succeeded", "data": {"router_model": router_model, "query_type": query_type}, "sessionId": "debug-session", "runId": "run1"}) + "\n")
        # #endregion
        
        # Validate response
        if query_type not in ["simple", "detailed", "research"]:
            query_type = "detailed"  # safer default if unclear
        
        return {"query_type": query_type, "needs_rag": query_type == "research"}
    
    
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
        """Handle detailed enzyme analysis queries."""
        # #region agent log
        import json as _json
        with open("/home/kellywang/project/protein_projects/protein-design-agent/.cursor/debug.log", "a") as _f:
            _f.write(_json.dumps({"hypothesisId": "H-C", "location": "agent.py:detailed_handler", "message": "entering detailed_handler", "data": {"worker_model": worker_model, "worker_temp": temperature}, "sessionId": "debug-session", "runId": "run1"}) + "\n")
        # #endregion
        
        llm_with_tools = worker_llm.bind_tools(tools)
        
        # Add skill instructions if not already present
        messages = state["messages"]
        
        # Check if first message is a system message
        has_system = (len(messages) > 0 and 
                     hasattr(messages[0], 'type') and 
                     messages[0].type == "system")
        
        if not has_system:
            system_msgs: list[SystemMessage] = []
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
            system_msgs.append(SystemMessage(content=DETAILED_ANALYSIS_SKILL))
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
    
    return workflow.compile()


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