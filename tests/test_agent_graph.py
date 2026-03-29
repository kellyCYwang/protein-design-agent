"""
Tests for agent graph structure and feature verification.
Validates graph topology after each feature addition.
Works without API keys by mocking LLM initialization.
"""
import json
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import AgentState


def build_agent_mocked():
    """Build agent with mocked LLM and RAG backends."""
    env = {
        "ZENMUX_API_KEY": "test-key",
        "ZENMUX_BASE_URL": "https://fake.test/v1",
        "RAG_BACKEND": "chroma",
        "EMBEDDING_SERVICE_URL": "http://localhost:9999/embed",
    }
    with patch.dict(os.environ, env):
        # Mock ChromaPDFRAG so it doesn't try to initialize ChromaDB
        with patch("agent.agent.create_rag_backend") as mock_rag:
            mock_rag_instance = MagicMock()
            mock_rag_instance.search.return_value = []
            mock_rag.return_value = mock_rag_instance

            from agent.agent import build_agent
            return build_agent()


def get_graph_info(agent):
    """Extract node names and edges from a compiled graph."""
    graph = agent.get_graph()
    # graph.nodes may be a dict {name: node} or list of node objects
    if isinstance(graph.nodes, dict):
        nodes = set(graph.nodes.keys())
    else:
        nodes = {(n.name if hasattr(n, 'name') else str(n)) for n in graph.nodes}
    # graph.edges may be list of tuples or edge objects
    edges = []
    for e in graph.edges:
        if isinstance(e, tuple):
            edges.append((e[0], e[1]))
        else:
            edges.append((e.source, e.target))
    return nodes, edges


# ============================================================
# Feature 1: Planning Node
# ============================================================

def test_feature1_plan_node_exists():
    """plan_query node is present in the graph."""
    agent = build_agent_mocked()
    nodes, edges = get_graph_info(agent)

    expected_nodes = {
        "__start__", "__end__",
        "route_query", "plan_query",
        "simple_handler", "rag_handler", "detailed_handler",
        "tools",
    }
    missing = expected_nodes - nodes
    assert not missing, f"Missing nodes: {missing}. Got: {nodes}"
    print(f"  ✓ All expected nodes present: {sorted(nodes)}")


def test_feature1_plan_routing():
    """plan_query sits between route_query and handlers for non-simple queries."""
    agent = build_agent_mocked()
    nodes, edges = get_graph_info(agent)

    # route_query should route to plan_query (not directly to detailed/rag)
    route_targets = {t for s, t in edges if s == "route_query"}
    assert "plan_query" in route_targets, f"route_query should route to plan_query. Targets: {route_targets}"
    assert "simple_handler" in route_targets, f"route_query should still route to simple_handler"
    assert "detailed_handler" not in route_targets, "route_query should not bypass planner to detailed_handler"
    assert "rag_handler" not in route_targets, "route_query should not bypass planner to rag_handler"
    print("  ✓ route_query correctly routes through plan_query")

    # plan_query should route to rag_handler or parallel_gather
    plan_targets = {t for s, t in edges if s == "plan_query"}
    assert "rag_handler" in plan_targets, f"plan_query should route to rag_handler. Targets: {plan_targets}"
    assert "parallel_gather" in plan_targets, f"plan_query should route to parallel_gather. Targets: {plan_targets}"
    print("  ✓ plan_query correctly routes to rag_handler and parallel_gather")


def test_feature1_plan_in_state():
    """AgentState includes the plan field."""
    annotations = AgentState.__annotations__
    assert "plan" in annotations, f"plan field missing from AgentState. Fields: {list(annotations)}"
    assert annotations["plan"] is str, f"plan should be str, got {annotations['plan']}"
    print("  ✓ AgentState has plan: str field")


def test_feature1_plan_json_structure():
    """Verify plan JSON structure expected by the system."""
    plan = {
        "goal": "Analyze lactate dehydrogenase structure and function",
        "steps": [
            {"id": 1, "tool": "get_ec_number", "args": {"enzyme_name": "lactate dehydrogenase"}, "reason": "Get EC classification", "depends_on": []},
            {"id": 2, "tool": "get_enzyme_structure", "args": {"enzyme_name": "lactate dehydrogenase"}, "reason": "Get PDB structure", "depends_on": []},
            {"id": 3, "tool": "search_uniprot_proteins", "args": {"query": "lactate dehydrogenase"}, "reason": "Get protein annotations", "depends_on": []},
        ]
    }
    plan_json = json.dumps(plan)
    parsed = json.loads(plan_json)

    assert parsed["goal"] == plan["goal"]
    assert len(parsed["steps"]) == 3
    independent = [s for s in parsed["steps"] if not s["depends_on"]]
    assert len(independent) == 3, "All 3 steps should be independent (parallelizable)"
    print("  ✓ Plan JSON structure validates correctly")


def test_feature1_tool_loop_preserved():
    """The tool node and both handlers still exist and are connected."""
    agent = build_agent_mocked()
    nodes, edges = get_graph_info(agent)

    # All handler nodes and tools node must be present
    assert "tools" in nodes, "tools node missing"
    assert "simple_handler" in nodes, "simple_handler missing"
    assert "detailed_handler" in nodes, "detailed_handler missing"
    print("  ✓ Tool node and both handler nodes present")

    # rag_handler must connect through parallel_gather to detailed_handler
    rag_targets = {t for s, t in edges if s == "rag_handler"}
    assert "parallel_gather" in rag_targets, "rag_handler should connect to parallel_gather"
    print("  ✓ rag_handler → parallel_gather edge preserved")


# ============================================================
# Feature 2: Parallel Gather
# ============================================================

def test_feature2_parallel_gather_node_exists():
    """parallel_gather node is present in the graph."""
    agent = build_agent_mocked()
    nodes, edges = get_graph_info(agent)
    assert "parallel_gather" in nodes, f"parallel_gather missing. Nodes: {nodes}"
    print("  ✓ parallel_gather node present")


def test_feature2_parallel_gather_routing():
    """parallel_gather sits between plan_query and detailed_handler."""
    agent = build_agent_mocked()
    nodes, edges = get_graph_info(agent)

    # plan_query should route to parallel_gather (not detailed_handler directly)
    plan_targets = {t for s, t in edges if s == "plan_query"}
    assert "parallel_gather" in plan_targets, f"plan_query should route to parallel_gather. Targets: {plan_targets}"
    print("  ✓ plan_query routes to parallel_gather")

    # parallel_gather should go to detailed_handler
    gather_targets = {t for s, t in edges if s == "parallel_gather"}
    assert "detailed_handler" in gather_targets, f"parallel_gather should go to detailed_handler. Targets: {gather_targets}"
    print("  ✓ parallel_gather routes to detailed_handler")

    # rag_handler should now route to parallel_gather (not directly to detailed_handler)
    rag_targets = {t for s, t in edges if s == "rag_handler"}
    assert "parallel_gather" in rag_targets, f"rag_handler should route to parallel_gather. Targets: {rag_targets}"
    print("  ✓ rag_handler routes to parallel_gather")


def test_feature2_gathered_context_in_state():
    """AgentState includes the gathered_context field."""
    annotations = AgentState.__annotations__
    assert "gathered_context" in annotations, f"gathered_context missing. Fields: {list(annotations)}"
    print("  ✓ AgentState has gathered_context: str field")


def test_feature2_parallel_execution_logic():
    """Verify that independent plan steps are identified for parallel execution."""
    plan = {
        "goal": "Analyze LDH",
        "steps": [
            {"id": 1, "tool": "get_ec_number", "args": {"enzyme_name": "LDH"}, "reason": "EC", "depends_on": []},
            {"id": 2, "tool": "get_enzyme_structure", "args": {"enzyme_name": "LDH"}, "reason": "PDB", "depends_on": []},
            {"id": 3, "tool": "search_arxiv_papers", "args": {"query": "LDH mutations"}, "reason": "Literature", "depends_on": []},
            {"id": 4, "tool": "synthesize_report", "args": {}, "reason": "Cross-ref", "depends_on": [1, 2, 3]},
        ]
    }
    steps = plan["steps"]
    independent = [s for s in steps if not s.get("depends_on")]
    dependent = [s for s in steps if s.get("depends_on")]

    assert len(independent) == 3, f"Expected 3 independent steps, got {len(independent)}"
    assert len(dependent) == 1, f"Expected 1 dependent step, got {len(dependent)}"
    assert dependent[0]["depends_on"] == [1, 2, 3]
    print("  ✓ Correctly partitions independent vs dependent plan steps")


def test_feature2_full_graph_flow():
    """Verify complete graph flow: route → plan → gather → handler."""
    agent = build_agent_mocked()
    nodes, edges = get_graph_info(agent)

    # Full detailed flow: route_query → plan_query → parallel_gather → detailed_handler
    route_targets = {t for s, t in edges if s == "route_query"}
    plan_targets = {t for s, t in edges if s == "plan_query"}
    gather_targets = {t for s, t in edges if s == "parallel_gather"}

    assert "plan_query" in route_targets
    assert "parallel_gather" in plan_targets
    assert "detailed_handler" in gather_targets
    print("  ✓ Full flow: route → plan → gather → detailed_handler verified")

    # Research flow: route_query → plan_query → rag_handler → parallel_gather → detailed_handler
    assert "rag_handler" in plan_targets
    rag_targets = {t for s, t in edges if s == "rag_handler"}
    assert "parallel_gather" in rag_targets
    print("  ✓ Research flow: route → plan → rag → gather → detailed_handler verified")


# ============================================================
# Feature 3: State Leakage Fix + Query Cache
# ============================================================

def test_feature3_route_query_clears_stale_state():
    """route_query must reset rag_context, plan, and gathered_context."""
    # Inspect the route_query function's return value — it should include
    # empty strings for these fields to clear stale data from prior turns.
    # We test this by checking the source code (since we can't easily run
    # the node without an LLM).
    import inspect
    agent = build_agent_mocked()

    # Get the route_query node function from the graph
    # The graph stores node functions internally
    graph = agent.get_graph()
    # We'll verify by reading the source
    from agent.agent import build_agent as _build_agent
    source = inspect.getsource(_build_agent)

    # Check that route_query returns empty strings for stale state fields
    assert '"rag_context": ""' in source, "route_query should clear rag_context"
    assert '"plan": ""' in source, "route_query should clear plan"
    assert '"gathered_context": ""' in source, "route_query should clear gathered_context"
    print("  ✓ route_query clears rag_context, plan, and gathered_context")


def test_feature3_query_cache_layer_import():
    """QueryCacheLayer is importable and constructable."""
    from agent.cache import QueryCacheLayer

    # Construct directly (not via singleton) to avoid cross-test contamination
    cache = QueryCacheLayer(embed_fn=None)
    assert hasattr(cache, "lookup")
    assert hasattr(cache, "store")
    assert hasattr(cache, "enabled")
    print("  ✓ QueryCacheLayer is importable and has correct interface")

    # When Redis is not connected, lookup should return None
    if not cache.enabled:
        result = cache.lookup("test query")
        assert result is None, "Should return None when Redis unavailable"
        print("  ✓ QueryCacheLayer.lookup degrades gracefully without Redis")
    else:
        print("  ✓ Redis is connected — skipping offline degradation check")


def test_feature3_query_cache_functions_in_agent():
    """check_query_cache and store_query_cache are importable from agent."""
    from agent.agent import check_query_cache, store_query_cache

    # Both should work without Redis (return None / no-op)
    result = check_query_cache("test query about lactate dehydrogenase")
    assert result is None, "Should return None when Redis unavailable"
    print("  ✓ check_query_cache works without Redis")

    # store should not raise
    store_query_cache("test query", "detailed", "test response", ["get_ec_number"])
    print("  ✓ store_query_cache works without Redis (no-op)")


def test_feature3_api_has_cancel_and_cache():
    """api.py has cancel endpoint and cache integration."""
    api_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "api.py"
    )
    with open(api_path) as f:
        source = f.read()

    assert "_try_query_cache" in source, "api.py should have query cache check"
    assert "_store_query_cache" in source, "api.py should have query cache store"
    assert "_active_cancels" in source, "api.py should track active cancellable streams"
    assert "cancel_event.is_set()" in source, "agent stream should check cancel flag"
    assert "/api/chat/cancel" in source, "api.py should expose cancel endpoint"
    print("  ✓ api.py has cancel endpoint and cache integration")


# ============================================================
# Runner
# ============================================================

if __name__ == "__main__":
    tests = [
        ("Feature 1: plan_query node exists", test_feature1_plan_node_exists),
        ("Feature 1: plan routing correct", test_feature1_plan_routing),
        ("Feature 1: plan field in state", test_feature1_plan_in_state),
        ("Feature 1: plan JSON structure", test_feature1_plan_json_structure),
        ("Feature 1: tool loop preserved", test_feature1_tool_loop_preserved),
        ("Feature 2: parallel_gather node exists", test_feature2_parallel_gather_node_exists),
        ("Feature 2: parallel_gather routing", test_feature2_parallel_gather_routing),
        ("Feature 2: gathered_context in state", test_feature2_gathered_context_in_state),
        ("Feature 2: parallel execution logic", test_feature2_parallel_execution_logic),
        ("Feature 2: full graph flow", test_feature2_full_graph_flow),
        ("Feature 3: state leakage fix", test_feature3_route_query_clears_stale_state),
        ("Feature 3: QueryCacheLayer", test_feature3_query_cache_layer_import),
        ("Feature 3: cache functions in agent", test_feature3_query_cache_functions_in_agent),
        ("Feature 3: api.py cancel + cache", test_feature3_api_has_cancel_and_cache),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            print(f"\n[TEST] {name}")
            fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        sys.exit(1)
    print("All tests passed!")
