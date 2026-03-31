get comprehensive enzyme information, including ec number, structure, catalytic mechanism, and catalytic residue analysis

# /skills/EnzymeAnalysis.md

## Workflow for Detailed Enzyme Analysis

When user asks "tell me more about enzyme X":

### Step 1: Gather comprehensive data
1. Call `get_ec_number(enzyme_name)` to get EC classification
2. Call `get_enzyme_structure(enzyme_name)` to get PDB info
3. Call `search_uniprot_proteins(enzyme_name)` to find UniProt entries
4. For the best-matching UniProt entry, call `get_uniprot_protein_details(uniprot_id)` to retrieve all relevant details including function, sequence, etc

### Step 2: Deep catalytic-residue analysis via literature
Search for papers that describe catalytic residues, active-site interactions, and mutagenesis studies:
1. Call `search_arxiv_papers("{enzyme_name}")` (max_results=5)
2. Call `search_preprints("{enzyme_name}")` (max_results=5)
3. From each returned paper abstract/excerpt, extract:
   - Residue numbers and identities (e.g. His57, Asp102, Ser195)
   - The role of each residue (nucleophile, general acid/base, oxyanion hole, transition-state stabilisation, etc.)
   - Known interactions (hydrogen bonds, salt bridges, covalent intermediates, metal coordination, substrate contacts)
   - Mutagenesis results if reported (e.g. "H57A → 10,000-fold loss of kcat")

### Step 3: Structure the response
Format as follows:
```
## Enzyme: {name}


**Example**:
- **Uniprot ID**: {uniprot_id}
- **Sequence**: {Sequence}
- **Organism**: {organism}
- **Function**: {function}

**EC Number**: {ec_number}

**Structure**:
- PDB ID: {pdb_id}
- Resolution: {resolution}
- UniProt ID: {uniprot_id}

**Catalytic Mechanism**:
{mechanism_details}

**Catalytic Residues**:
| Residue | Position | Role | Key Interactions | Evidence |
|---------|----------|------|------------------|----------|
| {residue_name} | {position} | {role} | {interactions} | {source: UniProt / PDB / literature} |
| ... | ... | ... | ... | ... |

**Interaction Network**:
- List each pair of residues that interact and the type of interaction
  (e.g. "Asp102 ── H-bond ── His57", "His57 ── covalent intermediate ── substrate")
- Note any metal ions or cofactors coordinated by catalytic residues

**Mutagenesis Highlights** (if available):
- {mutation} → {effect on activity} (Source: {reference})

**3D Visualization**:
[Link to structure viewer]
```

### Step 4: Quality checks
- Ensure all sections are populated
- If any API call fails, note it explicitly
- Cross-reference residue numbers between UniProt annotations and literature; flag any discrepancies
- Provide literature references (arXiv/bioRxiv URLs or DOIs) for every claim about residue roles