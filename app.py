import streamlit as st
import re
try:
    from stmol import showmol
    import py3Dmol
    HAS_STMOL = True
except ImportError:
    HAS_STMOL = False

from agent.agent import build_agent

# ============================================
# CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Protein Design Agent",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


def render_protein_structure(pdb_id: str):
    """Render protein structure using stmol/py3Dmol."""
    if not HAS_STMOL:
        st.warning("⚠️ `stmol` or `py3Dmol` not installed. Cannot visualize structure.")
        st.info("Try installing dependencies: `pip install stmol py3Dmol`")
        return

    try:
        # Create viewer
        viewer = py3Dmol.view(query=f'pdb:{pdb_id}')
        viewer.setStyle({'cartoon': {'color': 'spectrum'}})
        viewer.setBackgroundColor('white')
        
        # Add surface (optional, commented out for performance)
        # viewer.addSurface(py3Dmol.VDW, {'opacity':0.7,'color':'white'})
        
        viewer.zoomTo()
        
        # Render
        showmol(viewer, height=500, width=700)
    except Exception as e:
        st.error(f"Error rendering structure: {str(e)}")


def main():
    # Header
    st.title("🧬 Protein Design Agent")
    st.markdown("AI-powered enzyme analysis with intelligent query routing")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This agent can:
        - **Simple queries**: Get EC numbers quickly
        - **Detailed queries**: Comprehensive enzyme analysis
        
        The agent automatically detects query complexity and routes accordingly.
        """)
        
        st.divider()

        # In your sidebar section of app.py
        if st.button("🔄 Reload Agent"):
            if "agent" in st.session_state:
                del st.session_state.agent
            st.rerun()
        
        st.header("📋 Example Queries")
        
        if st.button("What's the EC number for chalcone isomerase?"):
            st.session_state.example_query = "What's the EC number for chalcone isomerase?"
        
        if st.button("Tell me about chalcone isomerase"):
            st.session_state.example_query = "Tell me about chalcone isomerase"
        
        if st.button("EC number for lysozyme"):
            st.session_state.example_query = "EC number for lysozyme"
        
        if st.button("Explain lactate dehydrogenase"):
            st.session_state.example_query = "Explain lactate dehydrogenase"
        
        if st.button("What is the model architecture of RFDiffusion?"):
            st.session_state.example_query = "What is the model architecture of RFDiffusion?"

        st.divider()
        
        st.header("🛠️ Available Tools")
        st.markdown("""
        - `get_ec_number()` - EC classification
        - `get_enzyme_structure()` - PDB data
        - `get_catalytic_mechanism()` - Reaction details
        """)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent" not in st.session_state:
        with st.spinner("Loading agent..."):
            st.session_state.agent = build_agent()
            
    # Use the new agent key
    agent = st.session_state.agent
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Get user input
    user_query = st.chat_input("Ask about an enzyme...")
    
    # Handle example query clicks
    if "example_query" in st.session_state:
        user_query = st.session_state.example_query
        del st.session_state.example_query
    
    if user_query:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Process with agent
        with st.chat_message("assistant"):
            # Create placeholders for dynamic updates
            status_placeholder = st.empty()
            progress_placeholder = st.empty()
            response_placeholder = st.empty()
            
            try:
                # Stream agent execution
                events = []
                final_response = None
                
                with st.spinner("Processing..."):
                    for event in agent.stream({
                        "messages": [("human", user_query)]
                    }):
                        events.append(event)
                        
                        # Update status based on event
                        if "route_query" in event:
                            query_type = event["route_query"].get("query_type", "unknown")
                            
                            if query_type == "simple":
                                status_placeholder.info("🔍 **Route**: Simple EC lookup")
                            else:
                                status_placeholder.info("📊 **Route**: Detailed analysis workflow")
                        
                        elif "read_skill" in event:
                            status_placeholder.success("📖 Loading analysis workflow...")
                        
                        elif "tools" in event:
                            tool_messages = event["tools"].get("messages", [])
                            if tool_messages:
                                tool_name = tool_messages[-1].name if hasattr(tool_messages[-1], 'name') else "unknown"
                                status_placeholder.success(f"🔧 Calling: `{tool_name}()`")
                        
                        elif "simple_handler" in event or "detailed_handler" in event:
                            node = "simple_handler" if "simple_handler" in event else "detailed_handler"
                            messages = event[node].get("messages", [])
                            if messages:
                                last_msg = messages[-1]
                                if hasattr(last_msg, 'content') and last_msg.content:
                                    final_response = last_msg.content
                
                # Display final response
                if final_response:
                    response_placeholder.markdown(final_response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_response
                    })
                    
                    # Check for PDB ID and render structure
                    pdb_match = re.search(r"PDB ID:\s*(\w{4})", final_response, re.IGNORECASE)
                    if pdb_match:
                        pdb_id = pdb_match.group(1)
                        with st.expander(f"🧬 3D Structure: {pdb_id}", expanded=True):
                            render_protein_structure(pdb_id)
                else:
                    response_placeholder.warning("No response generated")
                
                # Show execution summary
                with st.expander("🔍 Execution Trace", expanded=False):
                    st.json({
                        "total_steps": len(events),
                        "nodes_executed": list({k for e in events for k in e.keys()})
                    })
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)


if __name__ == "__main__":
    main()