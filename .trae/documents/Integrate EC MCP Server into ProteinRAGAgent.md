I will modify `agent/rag_query.py` to integrate the EC MCP Server into the `ProteinRAGAgent`.

### Implementation Steps

1. **Add Imports**:

   * Import `json` for tool argument parsing.

   * Import `sys` and `pathlib` to ensure correct path setup.

   * Import `ECMCPServer` from `mcp_servers.ec.server`.

2. **Update** **`ProteinRAGAgent.__init__`**:

   * Initialize `self.ec_server = ECMCPServer()`.

   * Initialize `self.tools` with EC-related tool definitions (lookup by EC number, search by product, class info).

3. **Add Tool Helper Methods**:

   * `_get_ec_tools(self)`: Returns the OpenAI-compatible tool schemas for the EC server.

   * `_execute_tool(self, tool_name, tool_args)`: Handles execution of EC tools.

4. **Refactor** **`query`** **Method**:

   * Maintain the existing RAG retrieval logic (getting context from papers).

   * Replace the single-turn `client.chat.completions.create` call with a **multi-turn loop**.

   * In the loop:

     * Send messages (including RAG context) and available `tools` to the model.

     * Handle `tool_calls` by executing the corresponding EC server method and appending the result to the conversation history.

     * Continue until the model provides a final answer or max iterations are reached.

   * Preserve the return format (`Dict` with answer, sources, etc.).

### Verification

* Run the `main` function in `rag_query.py`.

* The test prompt will be updated to ask a question that requires both RAG (papers) and EC lookup (e.g., "What is the reaction of EC 5.5.1.6 and are there papers about its engineering?").

