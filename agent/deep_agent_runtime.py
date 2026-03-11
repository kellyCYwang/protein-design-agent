"""
Deep Agents implementation for the protein design assistant.
"""

from functools import lru_cache
from typing import Any, Union
import os
import sys
import uuid

import dotenv
from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.rag.chroma_rag import ChromaPDFRAG
from agent.rag.pinecone_rag import PineconePDFRAG
from mcp_servers.arxiv.server import ArxivMCPServer
from mcp_servers.biorxiv.server import BiorxivMCPServer
from mcp_servers.ec.server import ECMCPServer
from mcp_servers.pdb.server import PdbMCPServer
from mcp_servers.uniprot.server import UniProtMCPServer

dotenv.load_dotenv()

ec_server = ECMCPServer()
pdb_server = PdbMCPServer()
arxiv_server = ArxivMCPServer()
biorxiv_server = BiorxivMCPServer()
uniprot_server = UniProtMCPServer()


class ZenmuxCompatibleChatOpenAI(ChatOpenAI):
    """ChatOpenAI variant that strips unsupported message names for Zenmux."""

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        for message in payload.get("messages", []):
            if isinstance(message, dict) and "name" in message:
                message.pop("name", None)
        return payload


@lru_cache(maxsize=256)
def _cached_ec_lookup(enzyme_name: str):
    normalized = enzyme_name.strip()
    return ec_server.get_ec_number_by_name(normalized)


@lru_cache(maxsize=256)
def _cached_structure_lookup(enzyme_name: str):
    normalized = enzyme_name.strip()
    return pdb_server.search_structure(normalized, max_results=1)


@tool
def get_ec_number(enzyme_name: str) -> str:
    """Get the EC number for an enzyme."""
    results = _cached_ec_lookup(enzyme_name)
    if results:
        best = results[0]
        enzyme_names = best.get("enzyme_names", [])
        name_str = enzyme_names[0] if enzyme_names else enzyme_name
        sources = ", ".join(best.get("sources", []))
        return f"EC {best['ec_number']} for {name_str} (Source: {sources})"
    return f"EC number not found for {enzyme_name}"


@tool
def get_enzyme_structure(enzyme_name: str) -> str:
    """Get 3D structure information for an enzyme."""
    results = _cached_structure_lookup(enzyme_name)
    if results and "error" not in results[0]:
        best = results[0]
        pdb_id = best.get("pdb_id", "Unknown")
        title = best.get("title", "No title")
        resolution = best.get("resolution", "N/A")
        method = best.get("method", "Unknown")
        organism = ", ".join(best.get("organism", []))
        return (
            f"PDB ID: {pdb_id}, Resolution: {resolution} Å, Organism: {organism}, "
            f"Method: {method}, Title: {title}"
        )
    return f"Structure not found for {enzyme_name}"


@tool
def search_arxiv_papers(query: str, max_results: int = 5) -> str:
    """Search arXiv for protein and enzyme engineering papers."""
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
        abstract = paper.get("abstract", "")
        if len(abstract) > 300:
            abstract = abstract[:300] + "..."
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
    """Search preprints for enzyme and protein design work."""
    results = biorxiv_server.integrated_search(query, max_results=max_results, source=source)
    if not results:
        return "No preprints found for this query."

    lines = []
    for i, paper in enumerate(results, 1):
        title = paper.get("title", "Unknown")
        authors = paper.get("authors", "Unknown")
        doi = paper.get("doi", "")
        pub_source = paper.get("source", "")
        pdb_id = paper.get("related_pdb", "")
        line = f"[{i}] {title}\n    Authors: {authors}\n    DOI: {doi}\n    Source: {pub_source}"
        if pdb_id:
            line += f"\n    Related PDB: {pdb_id}"
        lines.append(line + "\n")
    return "\n".join(lines).strip()


@tool
def search_uniprot_proteins(query: str, max_results: int = 5, organism: str | None = None) -> str:
    """Search UniProt for proteins by name, function, or keyword."""
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
        function = protein.get("function", "")
        if len(function) > 200:
            function = function[:200] + "..."
        sequence = protein.get("sequence", "")
        if len(sequence) > 60:
            sequence = sequence[:60] + "..."
        line = (
            f"[{i}] {name}\n"
            f"    UniProt ID: {uniprot_id}\n"
            f"    Organism: {org}\n"
            f"    Length: {length} aa"
        )
        if ec_nums:
            line += f"\n    EC: {ec_nums}"
        if function:
            line += f"\n    Function: {function}"
        line += f"\n    Sequence: {sequence}\n"
        lines.append(line)
    return "\n".join(lines).strip()


@tool
def get_uniprot_protein_details(uniprot_id: str) -> str:
    """Get detailed information for a specific UniProt entry."""
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
        sites_str = "; ".join(
            [f"Pos {site['position']}: {site['description']}" for site in active_sites[:5]]
        )
        lines.append(f"Active Sites: {sites_str}")
    pdb_ids = result.get("pdb_structures", [])
    if pdb_ids:
        lines.append(f"PDB Structures: {', '.join(pdb_ids)}")
    sequence = result.get("sequence", "")
    if sequence:
        suffix = "..." if len(sequence) > 100 else ""
        lines.append(f"Sequence: {sequence[:100]}{suffix}")
    return "\n".join(lines)


@tool
def search_enzyme_patents(query: str, max_results: int = 5) -> str:
    """Dummy patent stub used until the patent data source is implemented."""
    return (
        "Patent subagent is currently running in dummy mode. "
        f"No live patent lookup was performed for query: {query!r}. "
        "Use this response as a placeholder until a real patent backend is added."
    )


def _build_chat_model(model_name: str, temperature: float):
    """Build a chat model, preferring Zenmux and falling back to Anthropic."""
    api_key = os.getenv("ZENMUX_API_KEY")
    base_url = os.getenv("ZENMUX_BASE_URL", "https://api.zenmux.com/v1")
    if api_key and os.getenv("FORCE_ANTHROPIC", "false").lower() != "true":
        return ZenmuxCompatibleChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base=base_url,
        )

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_api_key:
        anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
        return ChatAnthropic(
            model=anthropic_model,
            temperature=temperature,
            anthropic_api_key=anthropic_api_key,
        )

    raise ValueError("No supported LLM provider configured: set ZENMUX_API_KEY or ANTHROPIC_API_KEY")


def create_rag_backend() -> Union[ChromaPDFRAG, PineconePDFRAG]:
    """Create the configured RAG backend."""
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
    print("Using Chroma RAG backend")
    return ChromaPDFRAG(
        pdf_directory=os.getenv("RAG_PDF_DIR", "./data/papers"),
        vd_persist_directory=os.getenv("RAG_CHROMA_DIR", "./data/chroma_db"),
        bm25_persist_directory=os.getenv("RAG_BM25_DIR", "./data/bm25"),
        embedding_service_url=os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8011/embed"),
    )


def _build_main_system_prompt() -> str:
    return """
You are the top-level orchestrator for a protein design and enzyme research assistant.

Use the built-in `task` tool to delegate specialized work to named subagents instead of
trying to do all domain analysis yourself.

Specialists:
- `sequence-structure-agent`: EC numbers, UniProt annotations, protein sequence facts,
  active sites, organism details, and PDB structure information.
- `literature-agent`: local indexed paper retrieval, method synthesis, arXiv, and preprints.
- `patent-agent`: enzyme patent discovery and high-level patent-landscape summaries.

Delegation policy:
- For simple EC/UniProt/PDB questions, delegate to `sequence-structure-agent`.
- For papers, recent methods, or literature reviews, delegate to `literature-agent`.
- For patent, IP, assignee, or filing questions, delegate to `patent-agent`.
- For comprehensive or "tell me everything" enzyme queries, break the request into
  independent domain subtasks and launch multiple `task` calls in the same turn.
- For comprehensive enzyme questions, use `sequence-structure-agent`, `literature-agent`,
  and `patent-agent` unless the user explicitly excludes one of those domains.
- Literature and patent work are independent, so launch those two concurrently whenever
  both are needed.

Subagents are stateless. Each task must include the full user context, the target enzyme
or protein, and the exact deliverable you want back.

After subagents return:
- Synthesize the findings into one coherent markdown answer.
- Separate factual findings from gaps, unavailable data, or tool failures.
- Never invent EC numbers, UniProt IDs, PDB IDs, patent numbers, assignees, or filing details.
""".strip()


def _build_subagents(search_research_papers, specialist_model: ChatOpenAI) -> list[dict[str, Any]]:
    return [
        {
            "name": "general-purpose",
            "description": (
                "Fallback agent for meta reasoning, coordination, or clarification when no domain "
                "specialist applies. Do not use for enzyme sequence, literature, or patent analysis."
            ),
            "system_prompt": (
                "You are a fallback subagent for generic coordination only. If the request is about "
                "enzyme sequence or structure, literature, or patents, the parent should use a "
                "specialist subagent instead."
            ),
            "tools": [],
            "model": specialist_model,
        },
        {
            "name": "sequence-structure-agent",
            "description": (
                "Handles enzyme and protein sequence, function, EC classification, UniProt data, "
                "active sites, and PDB structure lookup."
            ),
            "system_prompt": (
                "You are the sequence and structure specialist for proteins and enzymes. Use the "
                "available tools to ground your answer in EC, UniProt, and PDB data. Return concise "
                "markdown with summary, key molecular facts, sequence or annotation notes, structure "
                "notes, and gaps or uncertainties."
            ),
            "tools": [
                get_ec_number,
                get_enzyme_structure,
                search_uniprot_proteins,
                get_uniprot_protein_details,
            ],
            "model": specialist_model,
        },
        {
            "name": "literature-agent",
            "description": (
                "Handles scientific literature, local paper retrieval, method comparisons, arXiv, "
                "and enzyme-related preprints."
            ),
            "system_prompt": (
                "You are the literature specialist for protein and enzyme research. Prefer arXiv and "
                "preprint tools for general literature discovery, recent work, and broad enzyme "
                "summaries. Use the local paper RAG tool only when the user explicitly asks for local "
                "indexed papers, grounded excerpts from the local corpus, or a deep methods summary. "
                "Return concise markdown with literature summary, notable papers, method insights, and "
                "evidence gaps."
            ),
            "tools": [
                search_research_papers,
                search_arxiv_papers,
                search_preprints,
            ],
            "model": specialist_model,
        },
        {
            "name": "patent-agent",
            "description": (
                "Handles enzyme patent discovery and high-level patent-landscape summaries using "
                "heuristic public-web patent search."
            ),
            "system_prompt": (
                "You are the enzyme patent specialist. You must call the patent search tool before "
                "answering any patent discovery request or any comprehensive enzyme profile that asks "
                "for patents or IP coverage. Your only source of specific patent facts is the tool "
                "output. Do not answer from background knowledge. Do not invent patent numbers, "
                "assignees, filing dates, titles, or legal claims. If the tool fails or returns weak "
                "results, explicitly say that live patent retrieval was unsuccessful and stop there "
                "except for a brief caveat that the search is heuristic and not legal advice."
            ),
            "tools": [search_enzyme_patents],
            "model": specialist_model,
        },
    ]


def build_agent(model: str = "gpt-4o-mini", temperature: float = 0):
    """Build the protein design agent as an official Deep Agents orchestrator."""
    orchestrator_model_name = os.getenv("ROUTER_MODEL", os.getenv("LLM_MODEL", model))
    specialist_model_name = os.getenv("SUBAGENT_MODEL", os.getenv("ROUTER_MODEL", os.getenv("LLM_MODEL", model)))
    specialist_temperature = float(os.getenv("WORKER_TEMPERATURE", str(temperature)))

    orchestrator_model = _build_chat_model(orchestrator_model_name, 0)
    specialist_model = _build_chat_model(specialist_model_name, specialist_temperature)

    rag = create_rag_backend()
    print(f"Using RAG backend: {os.getenv('RAG_BACKEND', 'chroma').lower()}")

    @tool
    def search_research_papers(query: str, top_k: int = 5) -> str:
        """Search local indexed research papers for protein design and enzyme engineering insights."""
        try:
            results = rag.search(query, top_k=top_k)
        except Exception as exc:
            return f"Local paper search failed: {exc}"
        if not results:
            return "No relevant local papers found."

        lines: list[str] = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "") or "Unknown title"
            section = result.get("section", "") or "Unknown section"
            filename = result.get("filename", "") or "Unknown file"
            score = result.get("score", None)
            content = result.get("content", "") or ""
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

    return create_deep_agent(
        model=orchestrator_model,
        tools=[],
        system_prompt=_build_main_system_prompt(),
        subagents=_build_subagents(search_research_papers, specialist_model),
        checkpointer=MemorySaver(),
    )


def extract_final_response(final_state: dict[str, Any]) -> str | None:
    """Extract the last non-tool AI response from a final agent state."""
    for msg in reversed(final_state["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            return msg.content
    return None


def run_agent_sync(agent, query: str, thread_id: str | None = None) -> dict[str, Any]:
    """Run the agent synchronously and return response metadata."""
    final_state = agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": thread_id or f"sync-{uuid.uuid4()}"}},
    )
    return {
        "response": extract_final_response(final_state),
        "query_type": final_state.get("query_type"),
        "message_count": len(final_state["messages"]),
        "full_state": final_state,
    }
