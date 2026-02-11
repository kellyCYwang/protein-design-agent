#!/usr/bin/env python3
"""
MCP Server for EC (Enzyme Commission) number lookup
Queries BRENDA and ExPASy for enzyme classification and reaction information
"""

import requests
import time
from typing import Dict, List, Optional
import re
from bs4 import BeautifulSoup
from urllib.parse import quote
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class ECMCPServer:
    """MCP server for EC number and enzyme data"""
    
    EXPASY_BASE = "https://enzyme.expasy.org"
    BRENDA_BASE = "https://www.brenda-enzymes.org"
    
    def __init__(self):
        self.name = "ec_database"
        self.version = "1.0.0"
        self.session = self._create_retry_session()
        self.session.headers.update({
            'User-Agent': 'ECMCPServer/1.0'
        })
        self.rate_limit_delay = 0.5

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

    @staticmethod
    def _parse_expasy_enzyme_txt(content: str) -> Dict:
        records: Dict[str, List[str]] = {}
        current_key: Optional[str] = None

        for raw_line in content.splitlines():
            line = raw_line.rstrip()
            if not line:
                continue

            match = re.match(r"^([A-Z]{2})\s{3}(.*)$", line)
            if match:
                current_key = match.group(1)
                records.setdefault(current_key, []).append(match.group(2).strip())
                continue

            if current_key is not None and line.startswith(" "):
                records[current_key][-1] = f"{records[current_key][-1]} {line.strip()}"

        data: Dict[str, object] = {}

        accepted = records.get("DE", [])
        if accepted:
            data["accepted_name"] = accepted[0].strip().rstrip(".")
            data["name"] = data["accepted_name"]

        alt_names = records.get("AN", [])
        if alt_names:
            data["alternative_names"] = [n.strip().rstrip(".") for n in alt_names if n.strip()]

        reactions = records.get("CA", [])
        if reactions:
            data["reaction"] = " ".join(r.strip() for r in reactions if r.strip()).rstrip(".") + "."

        cofactors = records.get("CF", [])
        if cofactors:
            data["cofactors"] = [c.strip().rstrip(".") for c in cofactors if c.strip()]

        comments = records.get("CC", [])
        if comments:
            cleaned: List[str] = []
            for c in comments:
                cleaned.append(re.sub(r"^-!-\s*", "", c).strip())
            data["comments"] = [c for c in cleaned if c]

        dr_lines = records.get("DR", [])
        if dr_lines:
            refs: List[Dict[str, str]] = []
            for dr in dr_lines:
                for part in dr.split(";"):
                    part = part.strip()
                    if not part:
                        continue
                    m = re.match(r"^([A-Z0-9]{6,10}),\s*([A-Z0-9_]+)\s*$", part)
                    if not m:
                        continue
                    refs.append({"accession": m.group(1), "entry_name": m.group(2)})
            if refs:
                data["uniprot_references"] = refs

        return data

    @staticmethod
    def _parse_expasy_enzyme_html(content: str) -> Dict:
        soup = BeautifulSoup(content, "html.parser")
        data: Dict[str, object] = {}

        title = soup.find("h1")
        if title and title.get_text(strip=True):
            data["name"] = title.get_text(strip=True)

        return data
    
    # def lookup_ec_number(self, ec_number: str) -> Dict:
    #     """
    #     Look up enzyme information by EC number
        
    #     Args:
    #         ec_number: EC number (e.g., "5.5.1.6" for chalcone isomerase)
            
    #     Returns:
    #         Enzyme information including name, reactions, cofactors
    #     """
        
    #     # Clean EC number format
    #     ec_number = ec_number.strip().replace(" ", "")
        
    #     html_url = f"{self.EXPASY_BASE}/EC/{ec_number}"
    #     raw_url = f"{self.EXPASY_BASE}/EC/{ec_number}.txt"
        
    #     try:
    #         response = requests.get(raw_url, timeout=30)
    #         response.raise_for_status()
    #         content = response.text

    #         data = {
    #             "ec_number": ec_number,
    #             "url": html_url,
    #             "raw_url": raw_url,
    #         }

    #         data.update(self._parse_expasy_enzyme_txt(content))
            
    #         return data
            
    #     except requests.exceptions.HTTPError as e:
    #         if e.response.status_code == 404:
    #             return {"error": f"EC number {ec_number} not found"}
    #         return {"error": f"Failed to lookup EC number: {str(e)}"}
    #     except Exception as e:
    #         return {"error": f"Error looking up EC number: {str(e)}"}
    
    def search_uniprot(self, enzyme_name: str) -> List[Dict]:
        """Search UniProt for EC numbers"""
        results = []
        url = "https://rest.uniprot.org/uniprotkb/search"
        query = f'(protein_name:{enzyme_name}) AND (reviewed:true)'
        params = {
            "query": query,
            "format": "json",
            "size": 25,
            "fields": "accession,protein_name,ec,organism_name,gene_names"
        }
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            time.sleep(self.rate_limit_delay)
            
            if response.status_code == 200:
                data = response.json()
                ec_numbers_seen = set()
                
                for entry in data.get("results", []):
                    protein_desc = entry.get("proteinDescription", {})
                    rec_name = protein_desc.get("recommendedName", {})
                    ec_list = rec_name.get("ecNumbers", [])
                    
                    for ec_obj in ec_list:
                        ec = ec_obj.get("value")
                        if ec and ec not in ec_numbers_seen:
                            ec_numbers_seen.add(ec)
                            full_name = rec_name.get("fullName", {}).get("value", enzyme_name)
                            organism = entry.get("organism", {}).get("scientificName", "")
                            accession = entry.get("primaryAccession", "")
                            
                            results.append({
                                "ec_number": ec,
                                "enzyme_name": full_name,
                                "organism": organism,
                                "accession": accession,
                                "source": "UniProt",
                                "url": f"https://www.uniprot.org/uniprotkb/{accession}"
                            })
        except Exception as e:
            print(f"UniProt error: {e}")
        return results

    def search_kegg(self, enzyme_name: str) -> List[Dict]:
        """Search KEGG for EC numbers"""
        results = []
        try:
            url = f"https://rest.kegg.jp/find/enzyme/{quote(enzyme_name)}"
            response = self.session.get(url, timeout=15)
            time.sleep(self.rate_limit_delay)
            
            if response.status_code == 200 and response.text.strip():
                lines = response.text.strip().split("\n")
                for line in lines:
                    if "\t" in line:
                        enzyme_id, description = line.split("\t", 1)
                        ec = enzyme_id.replace("ec:", "")
                        results.append({
                            "ec_number": ec,
                            "enzyme_name": description.strip(),
                            "source": "KEGG",
                            "url": f"https://www.genome.jp/entry/ec:{ec}"
                        })
        except Exception as e:
            print(f"KEGG error: {e}")
        return results

    def search_rhea(self, enzyme_name: str) -> List[Dict]:
        """Search Rhea for EC numbers"""
        results = []
        try:
            url = "https://www.rhea-db.org/rest/1.0/ws/reaction/search"
            params = {"query": enzyme_name, "limit": 50}
            # Increased timeout to 30s
            response = self.session.get(url, params=params, timeout=30)
            time.sleep(self.rate_limit_delay)
            
            if response.status_code == 200:
                data = response.json()
                ec_seen = set()
                for result in data.get("results", []):
                    rhea_id = result.get("id")
                    detail_url = f"https://www.rhea-db.org/rest/1.0/ws/reaction/{rhea_id}"
                    # Increased timeout to 30s
                    detail_response = self.session.get(detail_url, timeout=30)
                    time.sleep(self.rate_limit_delay)
                    
                    if detail_response.status_code == 200:
                        detail = detail_response.json()
                        ec_list = detail.get("ec", [])
                        for ec_obj in ec_list:
                            ec = ec_obj.get("id")
                            ec_name = ec_obj.get("name", enzyme_name)
                            if ec and ec not in ec_seen:
                                ec_seen.add(ec)
                                results.append({
                                    "ec_number": ec,
                                    "enzyme_name": ec_name,
                                    "reaction_id": rhea_id,
                                    "source": "Rhea",
                                    "url": f"https://www.rhea-db.org/rhea/{rhea_id}"
                                })
        except Exception as e:
            # Log error but don't fail, just return empty list for Rhea
            print(f"Rhea search warning (non-fatal): {e}")
        return results

    def _merge_results(self, sources: Dict) -> List[Dict]:
        """Merge and deduplicate results from multiple sources"""
        ec_map = {}
        for source_name, results in sources.items():
            for result in results:
                ec = result["ec_number"]
                if ec not in ec_map:
                    ec_map[ec] = {
                        "ec_number": ec,
                        "enzyme_names": [],
                        "sources": [],
                        "organisms": set(),
                        "urls": []
                    }
                name = result.get("enzyme_name", "")
                if name and name not in ec_map[ec]["enzyme_names"]:
                    ec_map[ec]["enzyme_names"].append(name)
                if source_name not in ec_map[ec]["sources"]:
                    ec_map[ec]["sources"].append(source_name)
                if "organism" in result:
                    ec_map[ec]["organisms"].add(result["organism"])
                if "url" in result:
                    ec_map[ec]["urls"].append(result["url"])
        
        merged = []
        for ec, data in ec_map.items():
            data["organisms"] = list(data["organisms"])
            merged.append(data)
        merged.sort(key=lambda x: len(x["sources"]), reverse=True)
        return merged

    def _search_expasy_by_name(self, enzyme_name: str) -> List[Dict]:
        """Search ExPASy for EC numbers by enzyme name"""
        expasy_results = []
        search_url = f"{self.EXPASY_BASE}/cgi-bin/enzyme/enzyme-search-de"
        try:
            response = self.session.get(search_url, params={"Q": enzyme_name}, timeout=30)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                if "/EC/" in response.url and "enzyme-search" not in response.url:
                    ec = response.url.split("/EC/")[-1]
                    expasy_results.append({"ec_number": ec, "enzyme_name": enzyme_name, "source": "ExPASy", "url": response.url})
                else:
                    links = soup.find_all("a", href=re.compile(r"/EC/\d+\."))
                    for link in links:
                        ec = link.get_text().strip()
                        name_cell = link.find_next("td")
                        name = name_cell.get_text(strip=True) if name_cell else enzyme_name
                        if re.match(r"^\d+\.\d+\.\d+\.\d+$", ec):
                            expasy_results.append({"ec_number": ec, "enzyme_name": name, "source": "ExPASy", "url": f"{self.EXPASY_BASE}/EC/{ec}"})
        except Exception as e:
            print(f"ExPASy error: {e}")
        return expasy_results

    def get_ec_number_by_name(self, enzyme_name: str) -> List[Dict]:
        """
        Search for EC numbers by enzyme name using a cascade/fallback pattern.
        Tries sources in order: UniProt -> KEGG -> Rhea -> ExPASy.
        Returns as soon as one source finds results (fast path).
        
        Args:
            enzyme_name: Name of the enzyme to search for
            
        Returns:
            List of dictionaries containing EC numbers and names
        """
        # Cascade: try sources in order, return as soon as one succeeds
        sources_in_order = [
            ("UniProt", self.search_uniprot),
            ("KEGG", self.search_kegg),
            ("Rhea", self.search_rhea),
            ("ExPASy", self._search_expasy_by_name),
        ]
        
        for source_name, search_fn in sources_in_order:
            try:
                results = search_fn(enzyme_name)
                if results:
                    # Found results, return immediately (fast path)
                    return self._merge_results({source_name: results})
            except Exception as e:
                print(f"{source_name} failed: {e}, trying next source...")
                continue
        
        # All sources failed or returned nothing
        return []

    def search_by_reaction_product(self, product: str) -> List[Dict]:
        """
        Search for enzymes by reaction product
        
        Args:
            product: Product molecule name (e.g., "naringenin", "lactate")
            
        Returns:
            List of potential EC numbers and enzyme names
        """
        # For now, we'll use a simple mapping for common products
        # In production, you'd query BRENDA's database or use their SOAP API
        product_lower = product.lower()
        
        # Common enzyme-product mappings
        mappings = {
            "naringenin": [
                {"ec_number": "5.5.1.6", "enzyme_name": "Chalcone isomerase", "substrate": "chalcone"},
                {"ec_number": "1.14.14.82", "enzyme_name": "Flavanone 3-dioxygenase", "notes": "produces from naringenin"}
            ],
            "chalcone": [
                {"ec_number": "2.3.1.74", "enzyme_name": "Chalcone synthase", "product": "chalcone"},
            ],
            "lactate": [
                {"ec_number": "1.1.1.27", "enzyme_name": "L-lactate dehydrogenase", "product": "L-lactate"},
                {"ec_number": "1.1.1.28", "enzyme_name": "D-lactate dehydrogenase", "product": "D-lactate"},
            ],
            "ethanol": [
                {"ec_number": "1.1.1.1", "enzyme_name": "Alcohol dehydrogenase", "product": "ethanol"},
            ],
            "acetate": [
                {"ec_number": "6.2.1.1", "enzyme_name": "Acetate-CoA ligase", "substrate": "acetate"},
                {"ec_number": "2.3.1.8", "enzyme_name": "Phosphate acetyltransferase", "product": "acetate"},
            ],
        }
        
        results = []
        for key, enzymes in mappings.items():
            if product_lower in key or key in product_lower:
                results.extend(enzymes)
        
        if not results:
            return [{
                "message": f"No direct mapping found for '{product}'. Try searching bioRxiv/UniProt for enzymes producing this molecule.",
                "suggestion": f"Search UniProt with: 'enzyme {product}'"
            }]
        return results

    def get_enzyme_class_info(self, ec_class: str) -> Dict:
        """
        Get information about an enzyme class (first number of EC)
        
        Args:
            ec_class: EC class number (1-7)
                1 = Oxidoreductases
                2 = Transferases
                3 = Hydrolases
                4 = Lyases
                5 = Isomerases
                6 = Ligases
                7 = Translocases
                
        Returns:
            Information about the enzyme class
        """
        
        classes = {
            "1": {
                "name": "Oxidoreductases",
                "description": "Catalyze oxidation-reduction reactions",
                "examples": ["Dehydrogenases", "Oxidases", "Reductases"]
            },
            "2": {
                "name": "Transferases",
                "description": "Transfer functional groups between molecules",
                "examples": ["Methyltransferases", "Acyltransferases", "Kinases"]
            },
            "3": {
                "name": "Hydrolases",
                "description": "Catalyze hydrolysis reactions",
                "examples": ["Esterases", "Proteases", "Lipases", "Glycosidases"]
            },
            "4": {
                "name": "Lyases",
                "description": "Catalyze addition or removal of groups to form double bonds",
                "examples": ["Decarboxylases", "Aldolases", "Synthases"]
            },
            "5": {
                "name": "Isomerases",
                "description": "Catalyze isomerization reactions",
                "examples": ["Racemases", "Epimerases", "Isomerases"]
            },
            "6": {
                "name": "Ligases",
                "description": "Catalyze bond formation coupled with ATP hydrolysis",
                "examples": ["Synthetases", "Carboxylases"]
            },
            "7": {
                "name": "Translocases",
                "description": "Catalyze transmembrane transport",
                "examples": ["Transporters", "Channels"]
            }
        }
        
        return classes.get(ec_class, {"error": f"Invalid EC class: {ec_class}"})


def get_tools():
    """Return MCP tool definitions for EC database"""
    return [
        {
            "name": "lookup_ec_number",
            "description": "Look up enzyme information by EC number. Returns enzyme name, reaction, cofactors, and other details from the ExPASy database.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "ec_number": {
                        "type": "string",
                        "description": "EC number (e.g., '5.5.1.6' for chalcone isomerase, '3.1.1.1' for esterase)"
                    }
                },
                "required": ["ec_number"]
            }
        },
        {
            "name": "get_ec_number_by_name",
            "description": "Search for EC numbers by enzyme name.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "enzyme_name": {
                        "type": "string",
                        "description": "Name of the enzyme (e.g., 'chalcone isomerase')"
                    }
                },
                "required": ["enzyme_name"]
            }
        },
        {
            "name": "get_enzyme_class_info",
            "description": "Get information about an enzyme class (1=Oxidoreductases, 2=Transferases, 3=Hydrolases, 4=Lyases, 5=Isomerases, 6=Ligases, 7=Translocases)",
            "input_schema": {
                "type": "object",
                "properties": {
                    "ec_class": {
                        "type": "string",
                        "description": "EC class number (1-7)",
                        "enum": ["1", "2", "3", "4", "5", "6", "7"]
                    }
                },
                "required": ["ec_class"]
            }
        }
    ]


if __name__ == "__main__":
    # Test the EC server
    server = ECMCPServer()
    
    print("Testing EC Database MCP Server...\n")
    
    # Test 1: Lookup Chalcone Isomerase
    print("=" * 80)
    print("TEST 1: Lookup EC 5.5.1.6 (Chalcone Isomerase)")
    print("=" * 80)
    result = server.lookup_ec_number("5.5.1.6")
    
    if "error" not in result:
        print(f"\nEC Number: {result.get('ec_number')}")
        print(f"Name: {result.get('name', result.get('accepted_name'))}")
        if 'alternative_names' in result:
            print(f"Alternative names: {', '.join(result['alternative_names'])}")
        if 'reaction' in result:
            print(f"Reaction: {result['reaction']}")
        if 'cofactors' in result:
            print(f"Cofactors: {', '.join(result['cofactors'])}")
        print(f"URL: {result.get('url')}")
    else:
        print(f"Error: {result['error']}")
    
    # Test 2: Get enzyme class info
    print("\n" + "=" * 80)
    print("TEST 2: Get info about Hydrolases (EC class 3)")
    print("=" * 80)
    class_info = server.get_enzyme_class_info("3")
    
    print(f"\nClass: {class_info.get('name')}")
    print(f"Description: {class_info.get('description')}")
    print(f"Examples: {', '.join(class_info.get('examples', []))}")
