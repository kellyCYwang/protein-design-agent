#!/usr/bin/env python3
"""
MCP Server for RCSB PDB
Query protein structures and metadata
"""

import requests
from typing import List, Dict, Optional, Any

class PdbMCPServer:
    """MCP server for RCSB PDB queries"""
    
    DATA_API_URL = "https://data.rcsb.org/rest/v1/core/entry"
    POLYMER_API_URL = "https://data.rcsb.org/rest/v1/core/polymer_entity"
    SEARCH_API_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
    
    def __init__(self):
        self.name = "pdb"
        self.version = "1.0.0"
    
    def get_pdb_entry(self, pdb_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a PDB entry
        
        Args:
            pdb_id: PDB ID (e.g., "1CLL", "4HHB")
            
        Returns:
            Dictionary with structure metadata
        """
        pdb_id = pdb_id.upper()
        url = f"{self.DATA_API_URL}/{pdb_id}"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Extract relevant info
            info = {
                "pdb_id": data.get("rcsb_id"),
                "title": data.get("struct", {}).get("title", ""),
                "description": data.get("struct", {}).get("pdbx_descriptor", ""),
                "resolution": data.get("rcsb_entry_info", {}).get("resolution_combined", [None])[0],
                "polymer_composition": data.get("rcsb_entry_info", {}).get("polymer_composition", ""),
                "method": data.get("exptl", [{}])[0].get("method", ""),
                "deposition_date": data.get("rcsb_accession_info", {}).get("deposit_date", ""),
                "classification": data.get("struct_keywords", {}).get("pdbx_keywords", ""),
                "doi": data.get("rcsb_primary_citation", {}).get("pdbx_database_id_doi", ""),
                "organism": [],
                "expression_system": [],
                "mutation": "No",
                "macromolecules": []
            }

            print(info)
            
            # Fetch polymer entity details (assuming entity 1 is main)
            try:
                poly_url = f"{self.POLYMER_API_URL}/{pdb_id}/1"
                poly_response = requests.get(poly_url, timeout=10)
                if poly_response.status_code == 200:
                    poly_data = poly_response.json()
                    
                    # Organism
                    sources = poly_data.get("rcsb_entity_source_organism", [])
                    info["organism"] = [s.get("scientific_name") for s in sources if s.get("scientific_name")]
                    
                    # Expression System
                    hosts = poly_data.get("rcsb_entity_host_organism", [])
                    info["expression_system"] = [h.get("scientific_name") for h in hosts if h.get("scientific_name")]
                    
                    # Mutation
                    mutation_count = poly_data.get("rcsb_mutation_count", 0)
                    if mutation_count and mutation_count > 0:
                        info["mutation"] = f"Yes ({mutation_count})"
                    else:
                        info["mutation"] = "No"
            except Exception as e:
                print(f"Error fetching polymer details: {e}")
            
            return info
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return {"error": f"PDB ID {pdb_id} not found"}
            return {"error": f"Failed to fetch PDB data: {str(e)}"}
        except Exception as e:
            return {"error": f"An error occurred: {str(e)}"}

    def search_structure(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search for structures in PDB
        
        Args:
            query: Text query (e.g. "hemoglobin", "hydrolase")
            max_results: Max number of IDs to return
            
        Returns:
            List of PDB IDs with basic info
        """
        search_query = {
            "query": {
                "type": "terminal",
                "service": "full_text",
                "parameters": {
                    "value": query
                }
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {
                    "start": 0,
                    "rows": max_results
                }
            }
        }
        
        try:
            response = requests.post(self.SEARCH_API_URL, json=search_query, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            pdb_ids = [entry.get("identifier") for entry in data.get("result_set", [])]
            
            results = []
            for pid in pdb_ids:
                # Fetch details for each (could be slow, but fine for small N)
                details = self.get_pdb_entry(pid)
                if "error" not in details:
                    results.append(details)
                else:
                    results.append({"pdb_id": pid})
            
            return results
            
        except Exception as e:
            return [{"error": f"Search failed: {str(e)}"}]

def get_tools():
    """Return MCP tool definitions for PDB"""
    return [
        {
            "name": "get_pdb_entry",
            "description": "Get details about a protein structure from RCSB PDB by its 4-character ID. Returns title, resolution, method, and description.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pdb_id": {
                        "type": "string",
                        "description": "4-character PDB ID (e.g. '1CLL')"
                    }
                },
                "required": ["pdb_id"]
            }
        },
        {
            "name": "search_pdb",
            "description": "Search for protein structures in the PDB by keywords.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords to search (e.g. 'thermostable lipase')"
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    ]

if __name__ == "__main__":
    # Test
    server = PdbMCPServer()
    print("Fetching 1CLL...")
    print(server.get_pdb_entry("1CLL"))
    print("\nSearching for 'insulin'...")
    print(server.search_structure("insulin", max_results=3))
