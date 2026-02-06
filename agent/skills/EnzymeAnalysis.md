# /skills/EnzymeAnalysis.md

## Workflow for Detailed Enzyme Analysis

When user asks "tell me more about enzyme X":

### Step 1: Gather comprehensive data
1. Call `get_ec_number(enzyme_name)` to get EC classification
2. Call `get_enzyme_structure(enzyme_name)` to get PDB info
3. Call `get_catalytic_mechanism(ec_number)` with the EC from step 1

### Step 2: Structure the response
Format as follows:
```
## Enzyme: {name}

**EC Number**: {ec_number}

**Structure**:
- PDB ID: {pdb_id}
- Resolution: {resolution}

**Catalytic Mechanism**:
{mechanism_details}

**3D Visualization**:
[Link to structure viewer]
```

### Step 3: Quality checks
- Ensure all sections are populated
- If any API call fails, note it explicitly
- Provide literature references if available