I will implement the RCSB PDB integration as a new MCP Server and update the frontend to display 3D structures.

**Proposed Changes:**

1.  **Create PDB MCP Server (`mcp_servers/pdb/server.py`)**:
    *   Implement `PdbMCPServer` class to query RCSB PDB API.
    *   Add `get_pdb_entry(pdb_id)` to fetch metadata (Title, Resolution, Method, Polymer entities).
    *   Define `get_tools()` schema for the agent.

2.  **Update Agent (`agent/enzyme_agent.py`)**:
    *   Import and initialize `PdbMCPServer`.
    *   Register `get_pdb_entry` in `tools_map` and `_get_tools_schemas`.
    *   Update `AgentState` to include `pdb_hits`.
    *   Update `_tools_agent_node` to capture `pdb_hits` from tool results.
    *   Update system prompt to explicitly mention PDB capabilities.

3.  **Update Frontend (`app.py`)**:
    *   Modify `display_rich_content` to check for `pdb_hits`.
    *   Implement 3D visualization using `st.components.v1.html` with **3Dmol.js** (CDN version) to render the structure interactively when a PDB ID is found.

**Answer to your question:**
I will add it as a **Tool (MCP Server)**. This is the most robust way. I will also update the **prompt** in `enzyme_agent.py` to ensure the agent knows when to use this new tool.

**Files to Modify/Create:**
*   `mcp_servers/pdb/server.py` (Create)
*   `agent/enzyme_agent.py` (Modify)
*   `app.py` (Modify)

**Visualization Tech:**
I will use `3Dmol.js` via `st.components.v1.html` as it requires no extra Python dependencies and provides a great interactive experience.