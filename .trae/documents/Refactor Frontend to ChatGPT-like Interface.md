I will refactor the frontend in `app.py` to implement a ChatGPT-like interface.

**Proposed Changes:**

1. **Session State Management**:

   * Replace `query_history` with a standard `messages` list to store the conversation history in `[{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]` format.

   * Ensure `rag` and `agent` are initialized as before.

2. **Sidebar Updates**:

   * Add a primary "➕ New Chat" button at the top of the sidebar.

   * Clicking this will clear `st.session_state.messages` and generate a new `thread_id`.

3. **Main Chat Interface**:

   * Remove the "Chat" tab and make the chat interface the main view (keeping "Indexed Papers" and "About" in the sidebar and make it hiddenn default. &#x20;

   * **History Display**: Iterate through `st.session_state.messages` and display them using `st.chat_message()`.

     * For assistant messages, I will render the text answer and use `st.expander` to show sources and UniProt hits, keeping the rich display capabilities.

   * **Input Area**: Replace `st.text_area` and "Search" button with `st.chat_input("Ask a question...")` fixed at the bottom.

4. **Interaction Logic**:

   * When the user types in `st.chat_input`:

     1. Display the user's message immediately.
     2. Append it to `session_state.messages`.
     3. Display an empty assistant message block.
     4. Stream the agent's response into this block using the existing `agent.query_stream`.
     5. Once complete, append the full assistant response (including metadata like sources and hits) to `session_state.messages`.

5. **Migration**:

   * I will modify `app.py` directly. The existing logic for displaying results (UniProt hits, sources) will be moved into a helper function or directly inside the chat message rendering loop to ensure past messages also show this rich info.

**Files to Modify:**

* `app.py`

