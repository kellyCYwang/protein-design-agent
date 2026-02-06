# Analysis of Current State & LangGraph Migration

## 1. How the Agent Handles History for Different Users
Currently, your agent relies on **Streamlit's Session State** for user isolation.
*   **Mechanism**: Streamlit creates a separate `st.session_state` dictionary for each browser tab (user session).
*   **Isolation**: When User A opens the app, `st.session_state.agent` is a unique instance of `ProteinDesignAgent` with its own `self.conversation_history`. User B gets a completely separate instance.
*   **Limitation**: This memory is **ephemeral**. It lives only in the server's RAM. If the server restarts or the user closes the tab, the history is lost. It is not persisted to a database.

## 2. Can You Use LangChain / LangGraph?
**Yes, absolutely.** This is an ideal use case for **LangGraph**.
*   **Why**: Your current `query_stream` method manually implements a state machine:
    *   *State 1*: Try Tools (Loop).
    *   *Condition*: If answer found -> End. If "need info" -> Go to State 2.
    *   *State 2*: RAG Retrieval + Generation.
*   **LangGraph Fit**: This "Tools-First fallback" pattern is natively supported by LangGraph's conditional edges and state management.

## 3. Do You Need a Large Refactor?
**No, it is not a "large" refactor, but a "Medium" structural change.**
*   **What stays**: Your MCP tools (`server.py`), RAG logic (`enhanced_paper_rag.py`), and Streamlit UI (`app.py`) remain largely unchanged.
*   **What changes**: You would rewrite `enzyme_agent.py` to replace the manual `for` loops and `if/else` logic with a `StateGraph`.

---

# Proposed Migration Plan: LangGraph Refactor

If you wish to proceed, here is the plan to migrate the agent to LangGraph:

## Step 1: Define Agent State
Create a `TypedDict` state to track:
*   `messages`: List of LangChain `BaseMessage` objects (replaces `conversation_history`).
*   `question`: The user's input.
*   `retrieved_docs`: Documents found by RAG.
*   `rag_needed`: Boolean flag for the fallback decision.

## Step 2: Create Nodes
Refactor `enzyme_agent.py` methods into graph nodes:
*   **`tools_agent_node`**: Binds MCP tools to the LLM. Attempts to answer. Checks for "I need more info".
*   **`retrieve_node`**: Uses your existing `self.rag.search()` to fetch docs.
*   **`rag_agent_node`**: Calls LLM with the retrieved context (System Prompt v2).

## Step 3: Define Edges (The Logic)
Replace the manual `if` checks in `query_stream` with conditional edges:
*   From `tools_agent_node`:
    *   If tool calls present → Execute Tool.
    *   If answer final & valid → **END**.
    *   If answer is "I need info" → **Go to `retrieve_node`**.
*   From `retrieve_node` → `rag_agent_node` → **END**.

## Step 4: Integration
*   Update `ProteinDesignAgent` to compile and run this graph.
*   Update `query_stream` to stream events from the graph (`graph.stream()`).

## Effort Estimate
*   **Time**: ~30-60 minutes.
*   **Complexity**: Medium. The logic becomes much cleaner and easier to extend (e.g., adding a "Critique" step later).
