#!/usr/bin/env python3  
"""  
MCP Server for bioRxiv/medRxiv paper search  
Robust version: Handles Network Retries and Published vs. Preprint DOIs.  
"""  
  
import json  
import requests  
from requests.adapters import HTTPAdapter  
from urllib3.util.retry import Retry  
from typing import List, Dict, Optional, Any  
  
class BiorxivMCPServer:  
    """MCP server for searching bioRxiv papers via Europe PMC and RCSB"""  
      
    BIORXIV_API_URL = "https://api.biorxiv.org"  
    EPMC_API_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"  
    RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"  
    RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"  
      
    def __init__(self):  
        self.name = "biorxiv-enzyme-search"  
        self.version = "1.2.0"  
        # Initialize a session with retry logic for stability  
        self.session = self._create_retry_session()  
          
    def _create_retry_session(self):  
        """Creates a requests session that retries on connection errors."""  
        session = requests.Session()  
        retries = Retry(  
            total=3,   
            backoff_factor=1,   
            status_forcelist=[500, 502, 503, 504],  
            allowed_methods=["HEAD", "GET", "POST"]  
        )  
        adapter = HTTPAdapter(max_retries=retries)  
        session.mount("https://", adapter)  
        session.mount("http://", adapter)  
        return session  
  
    def is_biorxiv_doi(self, doi: str) -> bool:  
        """Checks if DOI belongs to bioRxiv/medRxiv (Preprint)."""  
        return doi.startswith("10.1101/")  
  
    def search_europe_pmc(self, query: str, max_results: int = 10) -> List[Dict]:  
        """  
        STRATEGY 1: Search Europe PMC for preprints using keywords.  
        """  
        print(f"[EuropePMC] Searching for: {query}")  
          
        search_params = {  
            "query": f"{query} AND (SRC:PPR)", # Restrict to Preprints  
            "format": "json",  
            "pageSize": max_results,  
            "resultType": "core",  
            "synonym": "true"   
        }  
          
        try:  
            # Use self.session for robustness  
            response = self.session.get(self.EPMC_API_URL, params=search_params, timeout=15)  
            response.raise_for_status()  
            data = response.json()  
              
            papers = []  
            for result in data.get("resultList", {}).get("result", []):  
                doi = result.get("doi", "")  
                papers.append({  
                    "title": result.get("title", ""),  
                    "authors": result.get("authorString", ""),  
                    "doi": doi,  
                    "source": result.get("source", "Preprint"),  
                    "published": result.get("firstPublicationDate", ""),  
                    "abstract": result.get("abstractText", "Abstract not available via search API."),  
                    "url": f"https://doi.org/{doi}" if doi else "",  
                    "search_method": "europe_pmc"  
                })  
            return papers  
        except Exception as e:  
            print(f"[EuropePMC] Error: {e}")  
            return []  
  
    def search_rcsb(self, query: str, max_results: int = 10) -> List[Dict]:  
        """  
        STRATEGY 2: Search RCSB PDB -> Get DOI (Preprint OR Published).  
        """  
        print(f"[RCSB] Searching PDB for structure keywords: {query}")  
          
        search_payload = {  
            "query": {  
                "type": "terminal",  
                "service": "full_text",  
                "parameters": {"value": query}  
            },  
            "return_type": "entry",  
            "request_options": {  
                "paginate": {"start": 0, "rows": max_results},  
                "results_content_type": ["experimental"]  
            }  
        }  
          
        try:  
            # Step A: Get PDB IDs  
            resp = self.session.post(self.RCSB_SEARCH_URL, json=search_payload, timeout=15)  
            if resp.status_code == 204:   
                return []  
            resp.raise_for_status()  
              
            pdb_ids = [entry["identifier"] for entry in resp.json().get("result_set", [])]  
            if not pdb_ids:  
                return []  
              
            # Step B: Get DOIs via GraphQL  
            graphql_query = """  
            query structure($ids: [String!]!) {  
              entries(entry_ids: $ids) {  
                rcsb_id  
                rcsb_primary_citation {  
                  pdbx_database_id_DOI  
                  title  
                  rcsb_authors  
                }  
              }  
            }  
            """  
              
            gql_resp = self.session.post(  
                self.RCSB_GRAPHQL_URL,   
                json={"query": graphql_query, "variables": {"ids": pdb_ids}},  
                timeout=15  
            )  
            gql_data = gql_resp.json()  
              
            papers = []  
            seen_dois = set()  
              
            for entry in gql_data.get("data", {}).get("entries", []):  
                citation = entry.get("rcsb_primary_citation")  
                if citation:  
                    doi = citation.get("pdbx_database_id_DOI")  
                    if doi and doi not in seen_dois:  
                        seen_dois.add(doi)  
                        papers.append({  
                            "title": citation.get("title", ""),  
                            "authors": ", ".join(citation.get("rcsb_authors", [])),  
                            "doi": doi,  
                            "related_pdb": entry.get("rcsb_id"),  
                            "source": "RCSB PDB",  
                            "search_method": "rcsb_pdb"  
                        })  
            return papers  
              
        except Exception as e:  
            print(f"[RCSB] Error: {e}")  
            return []  
  
    def fetch_crossref_details(self, doi: str) -> Dict:
        """
        Fallback: Fetch metadata from CrossRef for published papers.
        Useful when Europe PMC fails.
        """
        print(f"[CrossRef] Fetching details for {doi}...")
        url = f"https://api.crossref.org/works/{doi}"
        
        try:
            response = self.session.get(url, timeout=15)
            if response.status_code != 200:
                return {}
                
            data = response.json()
            item = data.get("message", {})
            
            # Extract authors
            authors_list = item.get("author", [])
            authors_str = ", ".join([f"{a.get('given','')} {a.get('family','')}".strip() for a in authors_list])
            
            # Extract date
            published = item.get("published-print", {}).get("date-parts", [[]])[0]
            if not published:
                 published = item.get("published-online", {}).get("date-parts", [[]])[0]
            date_str = "-".join(map(str, published)) if published else ""

            return {
                "title": item.get("title", [""])[0],
                "abstract": "Abstract not available via CrossRef metadata.", # CrossRef rarely has abstracts
                "authors": authors_str,
                "date": date_str,
                "journal": item.get("container-title", [""])[0],
                "url": f"https://doi.org/{doi}",
                "status": "published_crossref"
            }
        except Exception as e:
            print(f"[CrossRef] Error: {e}")
            return {}

    def find_preprint_for_published(self, title: str) -> Optional[Dict]:
        """
        Tries to find a preprint version of a published paper by searching its title in Europe PMC.
        """
        if not title:
            return None
            
        print(f"[Preprint Lookup] Searching for preprint of: {title[:50]}...")
        # Escape quotes in title
        safe_title = title.replace('"', '\\"')
        search_params = {
            "query": f'TITLE:"{safe_title}" AND SRC:PPR',
            "format": "json",
            "resultType": "core",
            "pageSize": 1
        }
        
        try:
            response = self.session.get(self.EPMC_API_URL, params=search_params, timeout=15)
            if response.status_code != 200:
                return None
                
            data = response.json()
            results = data.get("resultList", {}).get("result", [])
            
            if results:
                preprint = results[0]
                doi = preprint.get("doi")
                if doi:
                    return {
                        "preprint_doi": doi,
                        "preprint_url": f"https://doi.org/{doi}",
                        "preprint_abstract": preprint.get("abstractText", "")
                    }
            return None
        except Exception as e:
            print(f"[Preprint Lookup] Error: {e}")
            return None

    def get_paper_details(self, doi: str) -> Dict:
        """
        Smart Fetcher:
        1. Checks if DOI is bioRxiv (Preprint). If yes, gets full metadata/PDF.
        2. If DOI is Published, gets metadata from Europe PMC or CrossRef.
        3. Tries to find associated Preprint version for published papers.
        """
        if not doi:
            return {"error": "No DOI provided"}
            
        # --- PATH A: It's a bioRxiv Preprint ---
        if self.is_biorxiv_doi(doi):
            print(f"Fetching details from bioRxiv API for {doi}...")
            url = f"{self.BIORXIV_API_URL}/details/biorxiv/{doi}/na/json"
            try:
                response = self.session.get(url, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    if "collection" in data and len(data["collection"]) > 0:
                        paper = data["collection"][0]
                        return {
                            "title": paper.get("title", ""),
                            "abstract": paper.get("abstract", ""),
                            "authors": paper.get("authors", ""),
                            "date": paper.get("date", ""),
                            "version": paper.get("version", ""),
                            "category": paper.get("category", ""),
                            "pdf_url": f"https://www.biorxiv.org/content/{doi}v{paper.get('version','1')}.full.pdf",
                            "status": "preprint"
                        }
            except Exception as e:
                print(f"BioRxiv API failed: {e}")
        
        # --- PATH B: It's a Published Paper (or bioRxiv API failed) ---
        print(f"Fetching details from Europe PMC for published DOI {doi}...")
        
        paper_data = {}
        
        # B1. Try Europe PMC
        epmc_url = self.EPMC_API_URL
        params = {"query": f"DOI:{doi}", "format": "json", "resultType": "core"}
        try:
            response = self.session.get(epmc_url, params=params, timeout=15)
            data = response.json()
            result_list = data.get("resultList", {}).get("result", [])
            
            if result_list:
                res = result_list[0]
                paper_data = {
                    "title": res.get("title", ""),
                    "abstract": res.get("abstractText", "Abstract not available via public API."),
                    "authors": res.get("authorString", ""),
                    "date": res.get("firstPublicationDate", ""),
                    "journal": res.get("journalTitle", ""),
                    "url": f"https://doi.org/{doi}",
                    "status": "published"
                }
        except Exception as e:
            print(f"Europe PMC metadata fetch failed: {e}")

        # B2. Fallback to CrossRef if Europe PMC failed
        if not paper_data:
            paper_data = self.fetch_crossref_details(doi)
            
        if not paper_data:
            return {"error": "Paper details not found in public databases (EuropePMC, CrossRef)", "doi": doi}

        # --- PATH C: Try to find Preprint version for Published Paper ---
        # If we have a title, search for the preprint
        if paper_data.get("title"):
            preprint_info = self.find_preprint_for_published(paper_data["title"])
            if preprint_info:
                print(f"Found associated preprint: {preprint_info['preprint_doi']}")
                paper_data.update(preprint_info)
                
        return paper_data
  
    def integrated_search(self, query: str, max_results: int = 10, source: str = "all") -> List[Dict]:  
        results = []  
          
        # 1. Search Europe PMC (Best for text match)  
        if source in ["all", "biorxiv", "europepmc"]:  
            epmc_results = self.search_europe_pmc(query, max_results)  
            results.extend(epmc_results)  
              
        # 2. Search RCSB (Best for structural match)  
        if source in ["all", "pdb", "rcsb"] and len(results) < max_results:  
            rcsb_results = self.search_rcsb(query, max_results - len(results))  
            results.extend(rcsb_results)  
          
        # Deduplicate by DOI  
        unique_results = {}  
        for p in results:  
            if p.get('doi'):  
                unique_results[p['doi']] = p  
          
        return list(unique_results.values())[:max_results]  
  
# ==========================================  
# Tool Definitions for MCP  
# ==========================================  
  
def get_tools():  
    return [  
        {  
            "name": "search_enzyme_papers",  
            "description": "Search for enzyme/protein design papers using keywords. Searches both Europe PMC (preprints) and RCSB PDB (structures).",  
            "input_schema": {  
                "type": "object",  
                "properties": {  
                    "query": {  
                        "type": "string",  
                        "description": "Keywords like 'chalcone isomerase', 'de novo design'"  
                    },  
                    "max_results": {  
                        "type": "integer",  
                        "default": 10  
                    },  
                    "source": {  
                        "type": "string",  
                        "enum": ["all", "europepmc", "rcsb", "biorxiv"],  
                        "default": "all"  
                    }  
                },  
                "required": ["query"]  
            }  
        },  
        {  
            "name": "get_paper_details",  
            "description": "Get full abstract and metadata for a specific paper DOI. Handles both preprints and published papers.",  
            "input_schema": {  
                "type": "object",  
                "properties": {  
                    "doi": {"type": "string"}  
                },  
                "required": ["doi"]  
            }  
        }  
    ]  
  
# ==========================================  
# Testing Block  
# ==========================================  
  
if __name__ == "__main__":  
    server = BiorxivMCPServer()  
      
    print("--- Test 1: Search via Europe PMC (Direct Keyword) ---")  
    results = server.integrated_search("chalcone isomerase", max_results=2, source="europepmc")  
    for p in results:  
        print(f"[{p['search_method']}] {p['title']} (DOI: {p['doi']})")  
  
    print("\n--- Test 2: Search via RCSB (Indirect via Structure) ---")  
    # Searching for something that definitely has a published structure  
    results_rcsb = server.integrated_search("chalcone isomerase", max_results=5, source="rcsb")  
    for p in results_rcsb:  
        print(f"[{p['search_method']}] {p['title']} (PDB: {p.get('related_pdb')})")  
          
    print("\n--- Test 3: Get Details (Smart Fetcher) ---")  
      
    # A. Test with a Preprint DOI (if we found one in Test 1)  
    if results and server.is_biorxiv_doi(results[1]['doi']):  
        print(f"\n[Testing Preprint Logic] Fetching details for {results[1]['doi']}...")  
        details = server.get_paper_details(results[1]['doi'])  
        print(f"Status: {details.get('status')}")  
        print(f"Abstract: {details.get('abstract', '')[:100]}...")  
  

    # B. Test with a Published DOI (from RCSB result)  
    if results_rcsb:  
        doi_pub = results_rcsb[1]['doi']  
        print(f"\n[Testing Published Logic] Fetching details for {doi_pub}...")  
        details = server.get_paper_details(doi_pub)  
        print(f"Status: {details.get('status')}")  
        print(f"Journal: {details.get('journal', 'N/A')}")  
        print(f"Abstract: {details.get('abstract', '')[:100]}...")  

    # C. Test with the problematic DOI (CrossRef Fallback + Preprint Lookup)
    doi_problem = "10.1021/acscatal.9b01926"
    print(f"\n[Testing CrossRef Fallback & Preprint Lookup] Fetching details for {doi_problem}...")
    details = server.get_paper_details(doi_problem)
    print(f"Status: {details.get('status')}")
    print(f"Title: {details.get('title')}")
    print(f"Journal: {details.get('journal', 'N/A')}")
    print(f"Found Preprint DOI: {details.get('preprint_doi', 'None')}")
    print(f"Preprint Abstract: {details.get('preprint_abstract', '')[:100]}...")
