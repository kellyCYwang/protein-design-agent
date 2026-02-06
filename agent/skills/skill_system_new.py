"""
New skill system with TypedDict structure and progressive disclosure
Based on the refactor.md architecture plan
"""

from typing import TypedDict, List, Dict, Any, Optional
from abc import ABC, abstractmethod


class Skill(TypedDict):
    """A skill that can be progressively disclosed to the agent."""
    name: str  # Unique identifier for the skill
    description: str  # 1-2 sentence description to show in system prompt
    content: str  # Full skill content with detailed instructions


class BaseSkill(ABC):
    """Base class for all skills."""
    
    @property
    @abstractmethod
    def definition(self) -> Skill:
        """Return skill metadata as TypedDict."""
        pass
    
    @abstractmethod
    def execute(self, inputs: Dict[str, Any], context: str = "") -> Dict[str, Any]:
        """Execute the skill."""
        pass


class SkillMiddleware:
    """Middleware that injects skill descriptions into the system prompt."""
    
    def __init__(self):
        self.skills: Dict[str, Skill] = {}
        self.selected_skills: List[str] = []
    
    def register_skill(self, skill: BaseSkill):
        """Register a skill in the middleware."""
        skill_def = skill.definition()
        self.skills[skill_def["name"]] = skill_def
    
    def select_skills(self, skill_names: List[str]):
        """Select skills to be disclosed."""
        self.selected_skills = skill_names
    
    def inject_skills(self, system_prompt: str, selected_skills: List[str] = None) -> str:
        """Inject skill descriptions into the system prompt."""
        if selected_skills is None:
            selected_skills = self.selected_skills
        
        if not selected_skills:
            return system_prompt
        
        # Build skills section
        skills_section = "\n\n**AVAILABLE SKILLS:**\n"
        
        for skill_name in selected_skills:
            if skill_name in self.skills:
                skill = self.skills[skill_name]
                skills_section += f"\n**{skill['name']}**: {skill['description']}\n"
        
        skills_section += "\nTo use a skill, request it by name in your response."
        
        return system_prompt + skills_section
    
    def get_skill_content(self, skill_name: str) -> Optional[str]:
        """Get the full content of a specific skill."""
        if skill_name in self.skills:
            return self.skills[skill_name]["content"]
        return None
    
    def get_all_skills_description(self) -> str:
        """Get descriptions of all registered skills."""
        descriptions = []
        for skill_name, skill in self.skills.items():
            descriptions.append(f"**{skill_name}**: {skill['description']}")
        return "\n".join(descriptions)


# ==================== ANALYSIS SKILL IMPLEMENTATION ====================

class DetailedEnzymeAnalysisSkill(BaseSkill):
    """Detailed enzyme analysis skill with progressive disclosure."""
    
    def definition(self) -> Skill:
        return Skill(
            name="detailed_enzyme_analysis",
            description="Comprehensive enzyme characterization including structure, mechanism, and literature synthesis",
            content="""# Detailed Enzyme Analysis

You are an expert enzyme analyst. When this skill is activated, perform a comprehensive analysis following these steps:

## 1. Identity & Classification
- **EC Number**: Provide the complete EC classification
- **Enzyme Names**: List all accepted names and synonyms
- **Reaction**: Write the complete chemical reaction equation
- **Catalytic Mechanism**: Describe the step-by-step mechanism

## 2. Structural Analysis
- **Protein Structure**: Analyze 3D structure from PDB entries
- **Active Site**: Identify catalytic residues and their roles
- **Cofactors**: List required cofactors and their functions
- **Structural Features**: Note unique structural characteristics

## 3. Functional Properties
- **Substrate Specificity**: Describe substrate preferences
- **Kinetic Parameters**: Provide Km, kcat, and catalytic efficiency
- **pH Optimum**: Report optimal pH range
- **Temperature Stability**: Include thermal stability data

## 4. Biological Context
- **Metabolic Role**: Explain the enzyme's role in metabolism
- **Organism Distribution**: List organisms where found
- **Cellular Localization**: Describe subcellular location
- **Physiological Function**: Explain biological importance

## 5. Research Landscape
- **Recent Studies**: Summarize latest research findings
- **Engineering Applications**: Describe protein engineering efforts
- **Industrial Uses**: List biotechnological applications
- **Future Directions**: Suggest research opportunities

## Output Format
Present your analysis in a well-structured markdown format with:
- Clear headings and subheadings
- Bullet points for key information
- Tables for comparative data
- References to source databases (UniProt, RCSB PDB, BRENDA)

Always include hyperlinks to external databases for interactive exploration."""
        )
    
    def execute(self, inputs: Dict[str, Any], context: str = "") -> Dict[str, Any]:
        """Execute the detailed enzyme analysis."""
        
        enzyme_name = inputs.get("enzyme_name", "unknown enzyme")
        ec_number = inputs.get("ec_number", "")
        
        # For now, return a structured analysis based on available context
        # In a full implementation, this would query databases and perform analysis
        
        analysis = f"""# Detailed Analysis: {enzyme_name}

## Identity & Classification
- **EC Number**: {ec_number or "Not specified"}
- **Enzyme Name**: {enzyme_name}
- **Analysis Context**: {context}

## Structural Analysis
Based on available data, performing comprehensive structural characterization...

## Functional Properties
Analyzing kinetic parameters, substrate specificity, and optimal conditions...

## Research Landscape
Synthesizing recent literature and engineering applications...

*This is a placeholder implementation. Full analysis would integrate with UniProt, PDB, and literature databases.*"""
        
        return {
            "analysis_report": analysis,
            "enzyme_name": enzyme_name,
            "ec_number": ec_number,
            "status": "completed"
        }