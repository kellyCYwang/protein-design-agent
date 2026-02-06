#!/usr/bin/env python3
"""
MCP Server for UniProt database
Search for proteins by function, get sequences, find homologs
"""

import requests
from typing import List, Dict, Optional
import time

class UniProtMCPServer:
    """MCP server for UniProt database queries"""
    
    BASE_URL = "https://rest.uniprot.org"
    
    def __init__(self):
        self.name = "uniprot"
        self.version = "1.0.0"
    
    def search_proteins(
        self,
        query: str,
        max_results: int = 10,
        reviewed_only: bool = True,
        organism: Optional[str] = None
    ) -> List[Dict]:
        """
        Search UniProt for proteins
        
        Args:
            query: Search query (e.g., "chalcone isomerase",)
            max_results: Maximum results to return
            reviewed_only: Only return SwissProt (reviewed) entries
            organism: Filter by organism (e.g., "Thermus thermophilus", "Escherichia coli")
            
        Returns:
            List of protein entries with ID, name, organism, sequence
        """
        
        # Build query
        search_query = query
        if reviewed_only:
            search_query += " AND reviewed:true"
        if organism:
            search_query += f" AND organism_name:\"{organism}\""
        
        params = {
            "query": search_query,
            "format": "json",
            "size": max_results,
            "fields": "accession,id,protein_name,organism_name,length,sequence,ec,gene_names,cc_function,ft_act_site,ft_binding"
        }
        
        url = f"{self.BASE_URL}/uniprotkb/search"
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            
            proteins = []
            for entry in results:
                protein = {
                    "uniprot_id": entry.get("primaryAccession", ""),
                    "entry_name": entry.get("uniProtkbId", ""),
                    "protein_name": entry.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""),
                    "organism": entry.get("organism", {}).get("scientificName", ""),
                    "length": entry.get("sequence", {}).get("length", 0),
                    "sequence": entry.get("sequence", {}).get("value", ""),
                    "ec_numbers": [ec.get("value", "") for ec in entry.get("proteinDescription", {}).get("recommendedName", {}).get("ecNumbers", [])],
                    "gene_names": [gene.get("value", "") for gene in entry.get("genes", [])],
                }
                
                # Get function annotation
                comments = entry.get("comments", [])
                for comment in comments:
                    if comment.get("commentType") == "FUNCTION":
                        for text in comment.get("texts", []):
                            protein["function"] = text.get("value", "")
                            break
                
                proteins.append(protein)
            
            return proteins
            
        except Exception as e:
            return {"error": f"Failed to search UniProt: {str(e)}"}
    
    def get_protein_by_id(self, uniprot_id: str) -> Dict:
        """
        Get detailed protein information by UniProt ID
        
        Args:
            uniprot_id: UniProt accession (e.g., "P28583")
            
        Returns:
            Detailed protein information
        """
        
        url = f"{self.BASE_URL}/uniprotkb/{uniprot_id}.json"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            entry = response.json()
            
            protein = {
                "uniprot_id": entry.get("primaryAccession", ""),
                "entry_name": entry.get("uniProtkbId", ""),
                "protein_name": entry.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""),
                "organism": entry.get("organism", {}).get("scientificName", ""),
                "taxonomy_id": entry.get("organism", {}).get("taxonId", ""),
                "length": entry.get("sequence", {}).get("length", 0),
                "sequence": entry.get("sequence", {}).get("value", ""),
                "mass": entry.get("sequence", {}).get("molWeight", 0),
                "ec_numbers": [ec.get("value", "") for ec in entry.get("proteinDescription", {}).get("recommendedName", {}).get("ecNumbers", [])],
                "gene_names": [gene.get("value", "") for gene in entry.get("genes", [])],
            }
            
            # Get function
            comments = entry.get("comments", [])
            for comment in comments:
                if comment.get("commentType") == "FUNCTION":
                    for text in comment.get("texts", []):
                        protein["function"] = text.get("value", "")
                        break
            
            # Get features (active site, binding site)
            features = entry.get("features", [])
            active_sites = []
            binding_sites = []
            
            for feature in features:
                if feature.get("type") == "Active site":
                    active_sites.append({
                        "position": feature.get("location", {}).get("start", {}).get("value", ""),
                        "description": feature.get("description", "")
                    })
                elif feature.get("type") == "Binding site":
                    binding_sites.append({
                        "position": feature.get("location", {}).get("start", {}).get("value", ""),
                        "description": feature.get("description", "")
                    })
            
            if active_sites:
                protein["active_sites"] = active_sites
            if binding_sites:
                protein["binding_sites"] = binding_sites
            
            # Get PDB structures
            references = entry.get("uniProtKBCrossReferences", [])
            pdb_ids = []
            for ref in references:
                if ref.get("database") == "PDB":
                    pdb_ids.append(ref.get("id", ""))
            
            if pdb_ids:
                protein["pdb_structures"] = pdb_ids[:5]  # Limit to 5
            
            return protein
            
        except Exception as e:
            return {"error": f"Failed to get protein details: {str(e)}"}
    
    def search_by_ec_number(self, ec_number: str, max_results: int = 20) -> List[Dict]:
        """
        Search proteins by EC number
        
        Args:
            ec_number: EC number (e.g., "5.5.1.6")
            max_results: Maximum results
            
        Returns:
            List of proteins with that EC number
        """
        
        query = f"ec:{ec_number}"
        return self.search_proteins(query, max_results=max_results, reviewed_only=True)


def get_tools():
    """Return MCP tool definitions for UniProt"""
    return [
        {
            "name": "search_uniprot",
            "description": "Search UniProt for proteins by name, function, or keywords. Returns protein sequences, organism, EC numbers, and functional annotations.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'chalcone isomerase', 'thermostable lipase')"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (MUST be an integer, e.g. 10). Do NOT put organism name here.",
                        "default": 10
                    },
                    "organism": {
                        "type": "string",
                        "description": "Filter by organism name (e.g., 'Homo sapiens', 'Escherichia coli')"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "get_uniprot_protein",
            "description": "Get detailed information about a specific protein by UniProt ID, including sequence, active sites, and PDB structures.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "uniprot_id": {
                        "type": "string",
                        "description": "UniProt accession (e.g., 'P28583')"
                    }
                },
                "required": ["uniprot_id"]
            }
        },
        {
            "name": "search_uniprot_by_ec",
            "description": "Search proteins by EC number. Returns all proteins with that enzymatic activity.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "ec_number": {
                        "type": "string",
                        "description": "EC number (e.g., '5.5.1.6')"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results (default 20)",
                        "default": 20
                    }
                },
                "required": ["ec_number"]
            }
        },
    ]


if __name__ == "__main__":
    # Test UniProt server
    server = UniProtMCPServer()
    
    print("Testing UniProt MCP Server...\n")
    
    # Test 1: Search for chalcone isomerase
    print("=" * 80)
    print("TEST 1: Search for Chalcone Isomerase")
    print("=" * 80)
    results = server.search_proteins("chalcone isomerase", max_results=5)
    
    if isinstance(results, list):
        print(f"\nFound {len(results)} proteins:\n")
        for i, protein in enumerate(results, 1):
            print(f"{i}. {protein['protein_name']}")
            print(f"   UniProt ID: {protein['uniprot_id']}")
            print(f"   Organism: {protein['organism']}")
            print(f"   EC: {', '.join(protein['ec_numbers'])}")
            print(f"   Length: {protein['length']} aa")
            print(f"   Sequence: {protein['sequence'][:60]}...")
            if 'function' in protein:
                print(f"   Function: {protein['function'][:150]}...")
            print()
    else:
        print(f"Error: {results}")
    
    # Test 2: Search by EC number
    print("\n" + "=" * 80)
    print("TEST 2: Search by EC 5.5.1.6")
    print("=" * 80)
    results2 = server.search_by_ec_number("5.5.1.6", max_results=3)
    
    if isinstance(results2, list):
        print(f"\nFound {len(results2)} proteins:\n")
        for i, protein in enumerate(results2, 1):
            print(f"{i}. {protein['protein_name']} ({protein['organism']})")
            print(f"   UniProt ID: {protein['uniprot_id']}")
            print()

    
    # Test 4: Get detailed protein info
    if isinstance(results, list) and len(results) > 0:
        print("\n" + "=" * 80)
        print("TEST 4: Get detailed info for first protein")
        print("=" * 80)
        detailed = server.get_protein_by_id(results[0]['uniprot_id'])
        
        if "error" not in detailed:
            print(f"\nProtein: {detailed['protein_name']}")
            print(f"UniProt ID: {detailed['uniprot_id']}")
            print(f"Organism: {detailed['organism']}")
            print(f"Length: {detailed['length']} aa")
            print(f"Mass: {detailed.get('mass', 'N/A')} Da")
            if 'active_sites' in detailed:
                print(f"Active sites: {len(detailed['active_sites'])}")
                for site in detailed['active_sites'][:3]:
                    print(f"  - Position {site['position']}: {site['description']}")
            if 'pdb_structures' in detailed:
                print(f"PDB structures: {', '.join(detailed['pdb_structures'])}")

    
    # Test 5
    print("\n" + "=" * 80)
    print("TEST 5: Get amino acid sequence for Hexokinase-1 from Homo sapiens")
    print("=" * 80)
    sequence = server.search_proteins("Hexokinase-1", "Homo sapiens")
    
    if sequence:
        print(f"\nAmino Acid Sequence for Hexokinase-1:\n{sequence}")
    else:
        print("Failed to retrieve sequence")
