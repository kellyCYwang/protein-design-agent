"""
Protein Design Agent
~~~~~~~~~~~~~~~~~~~~
Compatibility wrapper for the official Deep Agents implementation.
"""

from agent.deep_agent_runtime import (
    build_agent,
    create_rag_backend,
    extract_final_response,
    get_ec_number,
    get_enzyme_structure,
    get_uniprot_protein_details,
    run_agent_sync,
    search_arxiv_papers,
    search_enzyme_patents,
    search_preprints,
    search_uniprot_proteins,
)


if __name__ == "__main__":
    agent = build_agent()

    test_queries = [
        "What's the EC number for chalcone isomerase?",
        "Search arXiv for RFDiffusion papers",
        "Find enzyme patents related to lipase detergents",
        "Give me comprehensive information about lactate dehydrogenase, including sequence, structure, papers, and patents.",
    ]

    for query in test_queries:
        print("=" * 80)
        print(f"QUERY: {query}")
        result = run_agent_sync(agent, query)
        print(f"Response: {result['response']}")
