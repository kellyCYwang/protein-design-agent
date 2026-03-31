Analyze the step-by-step catalytic mechanism of an enzyme, including chemical steps, transition states, and energetics

# Catalytic Mechanism Analysis

## When to activate
User asks about how an enzyme works, its reaction mechanism, catalytic cycle, chemical steps, or transition states. Examples:
- "How does chymotrypsin catalyze peptide bond cleavage?"
- "What is the catalytic mechanism of carbonic anhydrase?"
- "Explain the ping-pong mechanism of aspartate aminotransferase"
- "What are the chemical steps in lysozyme catalysis?"

## Workflow

### Step 1: Identify the enzyme and its reaction
1. Call `get_ec_number(enzyme_name)` — the EC class reveals the reaction type:
   - EC 1.x.x.x → oxidoreductase (electron transfer)
   - EC 2.x.x.x → transferase (group transfer)
   - EC 3.x.x.x → hydrolase (bond cleavage by water)
   - EC 4.x.x.x → lyase (non-hydrolytic bond cleavage)
   - EC 5.x.x.x → isomerase (intramolecular rearrangement)
   - EC 6.x.x.x → ligase (bond formation coupled to ATP)
2. Call `get_uniprot_protein_details(uniprot_id)` after finding the best entry via `search_uniprot_proteins(enzyme_name)` — extract:
   - Active site residues (positions and annotations)
   - Binding site residues
   - Cofactor requirements (metal ions, coenzymes, prosthetic groups)
   - The annotated catalytic activity / reaction string

### Step 2: Gather structural context
1. Call `get_enzyme_structure(enzyme_name)` to get the PDB entry
2. Note the resolution — higher resolution structures give more reliable active-site geometry
3. From UniProt annotations identify key catalytic residues and their spatial arrangement

### Step 3: Search literature for mechanistic detail
1. Call `search_arxiv_papers("{enzyme_name} catalytic mechanism")` (max_results=5)
2. Call `search_preprints("{enzyme_name} mechanism")` (max_results=5)
3. Call `search_research_papers("{enzyme_name} catalytic mechanism")` to check local indexed papers
4. From literature, extract:
   - Named mechanism type (e.g., ping-pong, ordered sequential, double-displacement, covalent catalysis)
   - Individual chemical steps with intermediates
   - Transition-state descriptions
   - Rate-limiting step if reported
   - pH dependence or kinetic parameters (kcat, Km) if available
   - Cofactor roles in the mechanism

### Step 4: Structure the response

Format the answer as follows:

```
## Catalytic Mechanism: {enzyme_name}

**Reaction**: {overall reaction equation, e.g., "peptide + H₂O → fragment₁ + fragment₂"}
**EC Number**: {ec_number}
**Mechanism Type**: {e.g., covalent catalysis, general acid-base, metal-ion catalysis, proximity/orientation}

---

### Active Site Architecture
- PDB ID: {pdb_id}
- Key residues: {list with positions, e.g., His57, Asp102, Ser195}
- Cofactors: {if any — Zn²⁺, NAD⁺, PLP, etc.}

### Step-by-Step Mechanism

**Step 1: {name, e.g., "Substrate binding"}**
- {Description of what happens chemically}
- Residues involved: {which residues participate and how}
- Intermediate formed: {if applicable}

**Step 2: {name, e.g., "Nucleophilic attack"}**
- {Description}
- Transition state: {geometry, charge distribution}
- Stabilized by: {oxyanion hole, metal ion, etc.}

**Step 3: {name, e.g., "Tetrahedral intermediate collapse"}**
- ...

*(Continue for all steps in the catalytic cycle)*

**Step N: {name, e.g., "Product release / enzyme regeneration"}**
- {How the enzyme returns to its resting state}

### Energy Profile
- Rate-limiting step: {which step and why}
- Catalytic acceleration: {fold-enhancement over uncatalyzed reaction, if known}
- Key catalytic strategies employed:
  - [ ] Covalent catalysis
  - [ ] General acid-base catalysis
  - [ ] Metal-ion catalysis
  - [ ] Electrostatic stabilization
  - [ ] Proximity and orientation effects
  - [ ] Transition-state stabilization

### Kinetic Parameters (if available)
| Parameter | Value | Conditions | Source |
|-----------|-------|------------|--------|
| kcat | {value} | {pH, temp} | {ref} |
| Km | {value} | {pH, temp} | {ref} |
| kcat/Km | {value} | — | {ref} |

### Inhibition & Pharmacological Relevance (if applicable)
- Known inhibitors that target the mechanism (e.g., mechanism-based inhibitors, transition-state analogs)
- Clinical relevance if the enzyme is a drug target

### References
- {Numbered list of papers cited, with DOIs or URLs}
```

### Step 5: Quality checks
- Every chemical step must specify which residue(s) act and their role (nucleophile, proton donor/acceptor, electrostatic stabilizer)
- Transition states should describe geometry (tetrahedral, planar, etc.) when known
- Distinguish between experimentally established steps and proposed/computational steps
- If the mechanism is debated in the literature, present both models and note the controversy
- Cross-check residue numbering between UniProt annotations and literature — flag discrepancies
- Cite sources for each mechanistic claim
