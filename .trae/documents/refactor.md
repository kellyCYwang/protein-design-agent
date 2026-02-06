# Enzyme Design Agent Architecture  
  
## Overview  
  
A LangGraph-based enzyme design agent with MCP server integrations, skills layer, and intelligent early termination for optimal response times.  
  

---  
  
## Architecture Overview  
┌─────────────────────────────────────────────────────────────────────────┐
│ AGENT LAYER │
│ (router, supervisor, reasoning) │
└─────────────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────────┐
│ SKILLS LAYER │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│ │ Active Site │ │ Mutation │ │Thermostability│ │ Substrate │ │
│ │ Analysis │ │ Impact │ │ Assessment │ │ Specificity │ │
│ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│ │ Homology │ │ Directed │ │ Docking │ │ Expression │ │
│ │ Search │ │ Evolution │ │ Protocol │ │ Optimization │ │
│ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────────┐
│ TOOLS LAYER (MCP Servers) │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │
│ │ EC │ │ RCSB │ │ BioRxiv │ │ RAG │ │ UniProt │ │
│ │ Lookup │ │ Query │ │ Search │ │ Retrieve│ │ Query │ │
│ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ │
└─────────────────────────────────────────────────────────────────────────┘

  
### Query Flow by Complexity  
  
| Query Type | Path | Response Time |  
|------------|------|---------------|  
| **Simple** | Router → Tools → END | ~1-2s |  
| **Moderate** | Router → Tools → BioRxiv → END | ~3-5s |  
| **Complex** | Router → Tools → Skills → Synthesizer → END | ~10-30s |  
  
---  
  
## State Definition  
  
```python  
from typing import TypedDict, List, Dict, Any, Literal, Annotated  
from langchain_core.messages import BaseMessage  
import operator  
  
  
class AgentState(TypedDict):  
    # Messages  
    messages: Annotated[List[BaseMessage], operator.add]  
    query: str  
    query_type: Literal["simple", "moderate", "complex"]  
      
    # Tool Results  
    tool_results: dict | None  
    biorxiv_results: List[dict] | None  
    retrieved_docs: List[str] | None  
      
    # Skills  
    selected_skills: List[str]  
    skill_inputs: Dict[str, Dict[str, Any]]  
    skill_results: Dict[str, Any]  
      
    # RAG  
    rag_results: dict | None  
      
    # Control Flags  
    is_complete: bool  
    next_steps: List[str]  
      
    # Output  
    final_response: str | None  
```

## Workflow Structure
Visual Flow with Early Termination
                    ┌─────────────────┐  
                    │   USER QUERY    │  
                    └────────┬────────┘  
                             │  
                             ▼  
                    ┌─────────────────┐  
                    │     ROUTER      │  
                    │                 │  
                    │ Classify:       │  
                    │ simple/moderate/│  
                    │ complex         │  
                    └────────┬────────┘  
                             │  
                             ▼  
                    ┌─────────────────┐  
                    │   TOOLS_AGENT   │  
                    │                 │  
                    │ Execute MCP     │  
                    │ tools           │  
                    └────────┬────────┘  
                             │  
              ┌──────────────┼──────────────┐  
              │              │              │  
              ▼              ▼              ▼  
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  
    │    END      │  │  BIORXIV    │  │   SKILL     │  
    │  (simple)   │  │  SEARCH     │  │  SELECTOR   │  
    │             │  │  (moderate) │  │  (complex)  │  
    │ Return      │  └──────┬──────┘  └──────┬──────┘  
    │ immediately │         │                │  
    └─────────────┘         │                ▼  
                            │       ┌─────────────┐  
                            │       │   SKILL     │  
                            │       │  EXECUTOR   │  
                            │       └──────┬──────┘  
                            │              │  
                            ▼              │  
                  ┌─────────────┐          │  
                  │    END      │          │  
                  │ (moderate)  │          │  
                  └─────────────┘          │  
                                           ▼  
                                 ┌─────────────────┐  
                                 │   SYNTHESIZER   │  
                                 └────────┬────────┘  
                                          │  
                                          ▼  
                                 ┌─────────────────┐  
                                 │      END        │  
                                 │   (complex)     │  
                                 └─────────────────┘  
## Workflow Builder

```python
def _build_workflow(self):  
    workflow = StateGraph(AgentState)  
      
    # Add Nodes  
    workflow.add_node("router", self._router_node)  
    workflow.add_node("tools_agent", self._tools_agent_node)  
    workflow.add_node("biorxiv_search", self._biorxiv_search_node)  
    workflow.add_node("retrieve", self._retrieve_node)  
    workflow.add_node("skill_selector", self._skill_selector_node)  
    workflow.add_node("skill_executor", self._skill_executor_node)  
    workflow.add_node("rag_agent", self._rag_agent_node)  
    workflow.add_node("synthesizer", self._synthesizer_node)  
      
    # Entry Point  
    workflow.add_edge(START, "router")  
    workflow.add_edge("router", "tools_agent")  
      
    # Tools Agent: Early exit for simple queries  
    workflow.add_conditional_edges(  
        "tools_agent",  
        self._should_continue_after_tools,  
        {  
            "END": END,  
            "biorxiv": "biorxiv_search",  
            "skills": "skill_selector",  
        }  
    )  
      
    # BioRxiv: Early exit for moderate queries  
    workflow.add_conditional_edges(  
        "biorxiv_search",  
        self._should_continue_after_biorxiv,  
        {  
            "END": END,  
            "retrieve": "retrieve",  
            "skills": "skill_selector",  
        }  
    )  
      
    # Retrieve  
    workflow.add_conditional_edges(  
        "retrieve",  
        self._should_continue_after_retrieve,  
        {  
            "skills": "skill_selector",  
            "synthesize": "synthesizer",  
        }  
    )  
      
    # Skills  
    workflow.add_edge("skill_selector", "skill_executor")  
    workflow.add_conditional_edges(  
        "skill_executor",  
        self._should_continue_after_skills,  
        {  
            "more_skills": "skill_executor",  
            "rag": "rag_agent",  
            "synthesize": "synthesizer",  
        }  
    )  
      
    # RAG Agent  
    workflow.add_edge("rag_agent", "synthesizer")  
      
    # Synthesizer always ends  
    workflow.add_edge("synthesizer", END)  
      
    return workflow.compile(checkpointer=MemorySaver())  
```


Skills Layer
Skill Base Class
```python
from abc import ABC, abstractmethod  
from dataclasses import dataclass  
from enum import Enum  
from typing import List, Dict, Any  
  
  
class SkillCategory(Enum):  
    ANALYSIS = "analysis"  
    DESIGN = "design"  
    SEARCH = "search"  
    PREDICTION = "prediction"  
    PROTOCOL = "protocol"  
  
  
@dataclass  
class SkillDefinition:  
    """Metadata about a skill for the agent to understand when to use it."""  
    name: str  
    description: str  
    category: SkillCategory  
    required_inputs: List[str]  
    outputs: List[str]  
    example_queries: List[str]  
    requires_rag: bool = True  
  
  
class BaseSkill(ABC):  
    """Base class for all skills."""  
      
    def __init__(self, llm, tools: Dict[str, Any], rag_retriever):  
        self.llm = llm  
        self.tools = tools  
        self.rag_retriever = rag_retriever  
      
    @property  
    @abstractmethod  
    def definition(self) -> SkillDefinition:  
        """Return skill metadata."""  
        pass  
      
    @abstractmethod  
    async def execute(self, inputs: Dict[str, Any], context: str = "") -> Dict[str, Any]:  
        """Execute the skill."""  
        pass  
```


Early Termination
Query Classification
```python
async def _router_node(self, state: AgentState) -> dict:  
    """Classify query complexity for routing."""  
      
    classification_prompt = """Classify this enzyme-related query by complexity:  
  
SIMPLE - Direct lookups, single fact retrieval:  
- "What's the EC number for lipase?"  
- "Get PDB structure 1ABC"  
- "What enzyme has EC 3.1.1.3?"  
  
MODERATE - Requires some context or multiple lookups:  
- "Find papers about lipase engineering"  
- "What's known about TEM-1 mutations?"  
- "Compare EC 3.1.1.3 and EC 3.1.1.4"  
  
COMPLEX - Requires analysis, skills, synthesis:  
- "Design thermostable variants of lipase"  
- "Analyze the active site and suggest mutations"  
- "Plan a directed evolution campaign"  
  
Query: {query}  
  
Respond with only: SIMPLE, MODERATE, or COMPLEX"""  
  
    response = await self.llm.ainvoke(  
        classification_prompt.format(query=state["query"])  
    )  
      
    query_type = response.content.strip().lower()  
      
    return {"query_type": query_type}  
```

## complete implementation
```python
from typing import TypedDict, List, Dict, Any, Literal, Annotated, Optional  
from langgraph.graph import StateGraph, END, START  
from langgraph.checkpoint.memory import MemorySaver  
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage  
from abc import ABC, abstractmethod  
from dataclasses import dataclass  
from enum import Enum  
import operator  
import json  
  
  
# ==================== STATE ====================  
  
class AgentState(TypedDict):  
    messages: Annotated[List[BaseMessage], operator.add]  
    query: str  
    query_type: Literal["simple", "moderate", "complex"]  
      
    # Results  
    tool_results: dict | None  
    biorxiv_results: List[dict] | None  
    retrieved_docs: List[str] | None  
      
    # Skills  
    selected_skills: List[str]  
    skill_inputs: Dict[str, Dict[str, Any]]  
    skill_results: Dict[str, Any]  
      
    # RAG  
    rag_results: dict | None  
      
    # Control  
    is_complete: bool  
    needs_retrieval: bool  
    next_steps: List[str]  
      
    # Output  
    final_response: str | None  
  
  
# ==================== SKILLS ====================  
  
class SkillCategory(Enum):  
    ANALYSIS = "analysis"  
    DESIGN = "design"  
    SEARCH = "search"  
    PREDICTION = "prediction"  
    PROTOCOL = "protocol"  
  
  
@dataclass  
class SkillDefinition:  
    name: str  
    description: str  
    category: SkillCategory  
    required_inputs: List[str]  
    outputs: List[str]  
    example_queries: List[str]  
    requires_rag: bool = True  
  
  
class BaseSkill(ABC):  
    def __init__(self, llm, tools: Dict[str, Any], rag_retriever):  
        self.llm = llm  
        self.tools = tools  
        self.rag_retriever = rag_retriever  
      
    @property  
    @abstractmethod  
    def definition(self) -> SkillDefinition:  
        pass  
      
    @abstractmethod  
    async def execute(self, inputs: Dict[str, Any], context: str = "") -> Dict[str, Any]:  
        pass  
  
  
# Include skill implementations here (ActiveSiteAnalysisSkill, etc.)  
# ... (as defined above)  
  
  
class SkillRegistry:  
    def __init__(self, llm, tools: Dict[str, Any], rag_retriever):  
        self.llm = llm  
        self.tools = tools  
        self.rag_retriever = rag_retriever  
        self.skills: Dict[str, BaseSkill] = {}  
        self._register_default_skills()  
      
    def _register_default_skills(self):  
        skill_classes = [  
            ActiveSiteAnalysisSkill,  
            MutationImpactSkill,  
            ThermostabilityAnalysisSkill,  
            DirectedEvolutionPlanningSkill,  
        ]  
        for skill_class in skill_classes:  
            skill = skill_class(self.llm, self.tools, self.rag_retriever)  
            self.skills[skill.definition.name] = skill  
      
    def register(self, skill: BaseSkill):  
        self.skills[skill.definition.name] = skill  
      
    def get_skill(self, name: str) -> Optional[BaseSkill]:  
        return self.skills.get(name)  
      
    def get_skills_prompt(self) -> str:  
        lines = ["Available Skills:\n"]  
        for skill in self.skills.values():  
            defn = skill.definition  
            lines.append(f"**{defn.name}** ({defn.category.value})")  
            lines.append(f"  Description: {defn.description}")  
            lines.append(f"  Inputs: {', '.join(defn.required_inputs)}")  
            lines.append(f"  Outputs: {', '.join(defn.outputs)}")  
            lines.append("")  
        return "\n".join(lines)  
  
  
# ==================== AGENT ====================  
  
class EnzymeDesignAgent:  
    def __init__(self, llm, mcp_tools: Dict[str, Any], rag_retriever):  
        self.llm = llm  
        self.mcp_tools = mcp_tools  
        self.rag_retriever = rag_retriever  
        self.skill_registry = SkillRegistry(llm, mcp_tools, rag_retriever)  
        self.workflow = self._build_workflow()  
      
    def _build_workflow(self):  
        workflow = StateGraph(AgentState)  
          
        # Nodes  
        workflow.add_node("router", self._router_node)  
        workflow.add_node("tools_agent", self._tools_agent_node)  
        workflow.add_node("biorxiv_search", self._biorxiv_search_node)  
        workflow.add_node("retrieve", self._retrieve_node)  
        workflow.add_node("skill_selector", self._skill_selector_node)  
        workflow.add_node("skill_executor", self._skill_executor_node)  
        workflow.add_node("rag_agent", self._rag_agent_node)  
        workflow.add_node("synthesizer", self._synthesizer_node)  
          
        # Edges  
        workflow.add_edge(START, "router")  
        workflow.add_edge("router", "tools_agent")  
          
        workflow.add_conditional_edges(  
            "tools_agent",  
            self._should_continue_after_tools,  
            {"END": END, "biorxiv": "biorxiv_search", "skills": "skill_selector"}  
        )  
          
        workflow.add_conditional_edges(  
            "biorxiv_search",  
            self._should_continue_after_biorxiv,  
            {"END": END, "retrieve": "retrieve", "skills": "skill_selector"}  
        )  
          
        workflow.add_conditional_edges(  
            "retrieve",  
            self._should_continue_after_retrieve,  
            {"skills": "skill_selector", "synthesize": "synthesizer"}  
        )  
          
        workflow.add_edge("skill_selector", "skill_executor")  
          
        workflow.add_conditional_edges(  
            "skill_executor",  
            self._should_continue_after_skills,  
            {"more_skills": "skill_executor", "rag": "rag_agent", "synthesize": "synthesizer"}  
        )  
          
        workflow.add_edge("rag_agent", "synthesizer")  
        workflow.add_edge("synthesizer", END)  
          
        return workflow.compile(checkpointer=MemorySaver())  
      
    # ==================== NODES ====================  
      
    async def _router_node(self, state: AgentState) -> dict:  
        """Classify query complexity."""  
        classification_prompt = """Classify this enzyme-related query by complexity:  
  
SIMPLE - Direct lookups, single fact retrieval:  
- "What's the EC number for lipase?"  
- "Get PDB structure 1ABC"  
  
MODERATE - Requires some context or multiple lookups:  
- "Find papers about lipase engineering"  
- "What's known about TEM-1 mutations?"  
  
COMPLEX - Requires analysis, skills, synthesis:  
- "Design thermostable variants of lipase"  
- "Analyze the active site and suggest mutations"  
  
Query: {query}  
  
Respond with only: SIMPLE, MODERATE, or COMPLEX"""  
  
        response = await self.llm.ainvoke(  
            classification_prompt.format(query=state["query"])  
        )  
        return {"query_type": response.content.strip().lower()}  
      
    async def _tools_agent_node(self, state: AgentState) -> dict:  
        """Execute MCP tools."""  
        tools = [  
            self.mcp_tools["query_ec_number"],  
            self.mcp_tools["query_rcsb"],  
            self.mcp_tools["get_protein_info"],  
        ]  
          
        llm_with_tools = self.llm.bind_tools(tools)  
          
        messages = [  
            {"role": "system", "content": "You are an enzyme database assistant. Answer directly and concisely."},  
            {"role": "user", "content": state["query"]}  
        ]  
          
        tool_results = {}  
        final_response = None  
          
        for _ in range(3):  
            response = await llm_with_tools.ainvoke(messages)  
            messages.append(response)  
              
            if not response.tool_calls:  
                final_response = response.content  
                break  
              
            for tool_call in response.tool_calls:  
                result = await self.mcp_tools[tool_call["name"]].ainvoke(tool_call["args"])  
                tool_results[tool_call["name"]] = result  
                messages.append({  
                    "role": "tool",  
                    "content": str(result),  
                    "tool_call_id": tool_call["id"]  
                })  
          
        # For simple queries, format response immediately  
        is_complete = False  
        if state["query_type"] == "simple" and tool_results:  
            is_complete = True  
            if not final_response:  
                format_response = await self.llm.ainvoke(  
                    f"Answer directly based on: {tool_results}\nQuery: {state['query']}"  
                )  
                final_response = format_response.content  
          
        return {  
            "tool_results": tool_results,  
            "final_response": final_response,  
            "is_complete": is_complete,  
            "messages": [AIMessage(content=final_response)] if final_response else [],  
        }  
      
    async def _biorxiv_search_node(self, state: AgentState) -> dict:  
        """Search bioRxiv for papers."""  
        tools = [  
            self.mcp_tools["biorxiv_search"],  
            self.mcp_tools["biorxiv_get_abstract"],  
        ]  
          
        llm_with_tools = self.llm.bind_tools(tools)  
          
        messages = [  
            {"role": "system", "content": "Search for relevant enzyme papers."},  
            {"role": "user", "content": f"Find papers about: {state['query']}"}  
        ]  
          
        biorxiv_results = []  
          
        for _ in range(3):  
            response = await llm_with_tools.ainvoke(messages)  
            messages.append(response)  
              
            if not response.tool_calls:  
                break  
              
            for tool_call in response.tool_calls:  
                result = await self.mcp_tools[tool_call["name"]].ainvoke(tool_call["args"])  
                biorxiv_results.append(result)  
                messages.append({  
                    "role": "tool",  
                    "content": str(result),  
                    "tool_call_id": tool_call["id"]  
                })  
          
        # For moderate queries, finish here  
        is_complete = False  
        final_response = None  
          
        if state["query_type"] == "moderate" and biorxiv_results:  
            is_complete = True  
            summary = await self.llm.ainvoke(  
                f"Summarize for query '{state['query']}':\nTools: {state.get('tool_results')}\nLiterature: {biorxiv_results}"  
            )  
            final_response = summary.content  
          
        needs_retrieval = "detailed" in state["query"].lower() or "mechanism" in state["query"].lower()  
          
        return {  
            "biorxiv_results": biorxiv_results,  
            "is_complete": is_complete,  
            "final_response": final_response or state.get("final_response"),  
            "needs_retrieval": needs_retrieval,  
        }  
      
    async def _retrieve_node(self, state: AgentState) -> dict:  
        """Retrieve full documents."""  
        docs = await self.rag_retriever.ainvoke(state["query"])  
        return {"retrieved_docs": [doc.page_content for doc in docs]}  
      
    async def _skill_selector_node(self, state: AgentState) -> dict:  
        """Select skills to execute."""  
        skills_description = self.skill_registry.get_skills_prompt()  
          
        selector_prompt = """Select skills for this enzyme design query.  
  
{skills_description}  
  
Query: {query}  
Available context: {context}  
  
Return JSON:  
{{  
    "selected_skills": ["skill_name_1", "skill_name_2"],  
    "skill_inputs": {{  
        "skill_name_1": {{"input_param": "value"}}  
    }}  
}}"""  
  
        response = await self.llm.ainvoke(  
            selector_prompt.format(  
                skills_description=skills_description,  
                query=state["query"],  
                context=f"Tools: {state.get('tool_results')}, Literature: {state.get('biorxiv_results')}"  
            )  
        )  
          
        try:  
            selection = json.loads(response.content)  
        except:  
            selection = {"selected_skills": [], "skill_inputs": {}}  
          
        return {  
            "selected_skills": selection.get("selected_skills", []),  
            "skill_inputs": selection.get("skill_inputs", {}),  
        }  
      
    async def _skill_executor_node(self, state: AgentState) -> dict:  
        """Execute selected skills."""  
        selected_skills = state.get("selected_skills", [])  
        skill_inputs = state.get("skill_inputs", {})  
        skill_results = state.get("skill_results", {}) or {}  
          
        context = f"Tools: {state.get('tool_results')}\nLiterature: {state.get('biorxiv_results')}"  
          
        for skill_name in selected_skills:  
            if skill_name in skill_results:  
                continue  
              
            skill = self.skill_registry.get_skill(skill_name)  
            if skill:  
                inputs = skill_inputs.get(skill_name, {})  
                result = await skill.execute(inputs, context)  
                skill_results[skill_name] = result  
          
        remaining_skills = [s for s in selected_skills if s not in skill_results]  
          
        return {  
            "skill_results": skill_results,  
            "selected_skills": remaining_skills,  
        }  
      
    async def _rag_agent_node(self, state: AgentState) -> dict:  
        """Deep RAG analysis."""  
        docs = await self.rag_retriever.ainvoke(state["query"])  
        rag_context = "\n".join([doc.page_content for doc in docs[:5]])  
          
        analysis_prompt = """Expert enzyme design analysis.  
  
Query: {query}  
Tool Results: {tools}  
Literature: {literature}  
Skill Results: {skills}  
Additional Literature: {rag}  
  
Provide comprehensive analysis."""  
  
        response = await self.llm.ainvoke(  
            analysis_prompt.format(  
                query=state["query"],  
                tools=state.get("tool_results"),  
                literature=state.get("biorxiv_results"),  
                skills=state.get("skill_results"),  
                rag=rag_context  
            )  
        )  
          
        return {"rag_results": {"analysis": response.content}}  
      
    async def _synthesizer_node(self, state: AgentState) -> dict:  
        """Synthesize final response."""  
        skill_results_formatted = ""  
        for name, result in (state.get("skill_results") or {}).items():  
            skill_results_formatted += f"\n### {name}\n{result.get('analysis', result)}\n"  
          
        synthesis_prompt = """Synthesize a comprehensive enzyme design response.  
  
Query: {query}  
Database Results: {tools}  
Literature: {literature}  
Skill Analyses: {skills}  
RAG Analysis: {rag}  
  
Create a well-structured, actionable response."""  
  
        response = await self.llm.ainvoke(  
            synthesis_prompt.format(  
                query=state["query"],  
                tools=state.get("tool_results", "N/A"),  
                literature=state.get("biorxiv_results", "N/A"),  
                skills=skill_results_formatted or "N/A",  
                rag=state.get("rag_results", "N/A"),  
            )  
        )  
          
        return {  
            "final_response": response.content,  
            "messages": [AIMessage(content=response.content)]  
        }  
      
    # ==================== ROUTING ====================  
      
    def _should_continue_after_tools(self, state: AgentState) -> str:  
        if state.get("is_complete"):  
            return "END"  
        query_type = state.get("query_type", "simple")  
        if query_type == "simple":  
            return "END"  
        elif query_type == "moderate":  
            return "biorxiv"  
        return "skills"  
      
    def _should_continue_after_biorxiv(self, state: AgentState) -> str:  
        if state.get("is_complete"):  
            return "END"  
        if state.get("query_type") == "moderate":  
            return "END"  
        if state.get("needs_retrieval"):  
            return "retrieve"  
        return "skills"  
      
    def _should_continue_after_retrieve(self, state: AgentState) -> str:  
        if state.get("query_type") == "complex":  
            return "skills"  
        return "synthesize"  
      
    def _should_continue_after_skills(self, state: AgentState) -> str:  
        if state.get("selected_skills"):  
            return "more_skills"  
        skill_results = state.get("skill_results", {})  
        for result in skill_results.values():  
            if result.get("needs_more_rag"):  
                return "rag"  
        return "synthesize"  
      
    # ==================== RUN ====================  
      
    async def run(self, query: str, thread_id: str = "default") -> str:  
        """Execute the workflow."""  
        initial_state = {  
            "messages": [HumanMessage(content=query)],  
            "query": query,  
            "query_type": "simple",  
            "tool_results": None,  
            "biorxiv_results": None,  
            "retrieved_docs": None,  
            "selected_skills": [],  
            "skill_inputs": {},  
            "skill_results": {},  
            "rag_results": None,  
            "is_complete": False,  
            "needs_retrieval": False,  
            "next_steps": [],  
            "final_response": None,  
        }  
          
        config = {"configurable": {"thread_id": thread_id}}  
        final_state = await self.workflow.ainvoke(initial_state, config)  
          
        return final_state["final_response"]  
```