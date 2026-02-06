#!/usr/bin/env python3
"""
MCP Server for arXiv paper search
Focuses on protein/enzyme engineering papers
"""

import json
import requests
import xmltodict
from typing import List, Dict, Optional
from datetime import datetime

class ArxivMCPServer:
    """MCP server for searching arXiv papers"""
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    def __init__(self):
        self.name = "arxiv"
        self.version = "1.0.0"
        
    def search_papers(
        self, 
        query: str, 
        max_results: int = 10,
        categories: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Search arXiv for papers
        
        Args:
            query: Search query (e.g., "chalcone isomerase enzyme engineering")
            max_results: Maximum number of results to return
            categories: List of arXiv categories (default: q-bio, cs.LG)
            
        Returns:
            List of papers with title, authors, abstract, url, published date
        """
        
        # Default to biology and ML categories
        if categories is None:
            categories = ["q-bio", "cs.LG", "cs.AI"]
        
        # Build query with categories
        category_query = " OR ".join([f"cat:{cat}" for cat in categories])
        full_query = f"({query}) AND ({category_query})"
        
        params = {
            "search_query": full_query,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            data = xmltodict.parse(response.content)
            
            # Handle single vs multiple entries
            entries = data.get("feed", {}).get("entry", [])
            if not isinstance(entries, list):
                entries = [entries] if entries else []
            
            papers = []
            for entry in entries:
                # Extract authors
                authors = entry.get("author", [])
                if not isinstance(authors, list):
                    authors = [authors]
                author_names = [a.get("name", "") for a in authors]
                
                # Extract published date
                published = entry.get("published", "")
                try:
                    pub_date = datetime.fromisoformat(published.replace('Z', '+00:00'))
                    published_str = pub_date.strftime("%Y-%m-%d")
                except:
                    published_str = published[:10] if len(published) >= 10 else ""
                
                paper = {
                    "title": entry.get("title", "").replace("\n", " ").strip(),
                    "authors": author_names,
                    "abstract": entry.get("summary", "").replace("\n", " ").strip(),
                    "url": entry.get("id", ""),
                    "published": published_str,
                    "arxiv_id": entry.get("id", "").split("/")[-1],
                }
                papers.append(paper)
            
            return papers
            
        except Exception as e:
            return {
                "error": f"Failed to search arXiv: {str(e)}"
            }
    
    def get_paper_details(self, arxiv_id: str) -> Dict:
        """
        Get full details of a specific paper by arXiv ID
        
        Args:
            arxiv_id: arXiv identifier (e.g., "2301.12345")
            
        Returns:
            Paper details
        """
        params = {
            "id_list": arxiv_id,
            "max_results": 1
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            data = xmltodict.parse(response.content)
            entry = data.get("feed", {}).get("entry", {})
            
            if not entry:
                return {"error": f"Paper {arxiv_id} not found"}
            
            authors = entry.get("author", [])
            if not isinstance(authors, list):
                authors = [authors]
            
            return {
                "title": entry.get("title", "").replace("\n", " ").strip(),
                "authors": [a.get("name", "") for a in authors],
                "abstract": entry.get("summary", "").replace("\n", " ").strip(),
                "url": entry.get("id", ""),
                "pdf_url": entry.get("id", "").replace("/abs/", "/pdf/") + ".pdf",
                "published": entry.get("published", "")[:10],
                "categories": entry.get("category", []),
            }
            
        except Exception as e:
            return {"error": f"Failed to get paper details: {str(e)}"}


# MCP Interface (simplified - we'll integrate with Claude later)
def get_tools():
    """Return MCP tool definitions"""
    return [
        {
            "name": "search_arxiv",
            "description": "Search arXiv for scientific papers on protein/enzyme engineering, machine learning for biology, or related topics. Returns paper titles, abstracts, authors, and URLs.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'chalcone isomerase', 'thermostable enzymes', 'protein design machine learning')"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 10)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "get_paper_details",
            "description": "Get full details of a specific arXiv paper by its ID",
            "input_schema": {
                "type": "object",
                "properties": {
                    "arxiv_id": {
                        "type": "string",
                        "description": "arXiv paper ID (e.g., '2301.12345')"
                    }
                },
                "required": ["arxiv_id"]
            }
        }
    ]


if __name__ == "__main__":
    # Test the server
    server = ArxivMCPServer()
    
    print("Testing arXiv MCP Server...\n")
    
    # Test search
    print("=" * 80)
    print("TEST 1: Search for RFdiffusion papers")
    print("=" * 80)
    results = server.search_papers("RFdiffusion", max_results=5)
    
    if isinstance(results, list):
        print(f"\nFound {len(results)} papers:\n")
        for i, paper in enumerate(results, 1):
            print(f"{i}. {paper['title']}")
            print(f"   Authors: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}")
            print(f"   Published: {paper['published']}")
            print(f"   URL: {paper['url']}")
            print(f"   Abstract: {paper['abstract'][:200]}...")
            print()
    else:
        print(f"Error: {results}")
    
    # Test broader enzyme engineering search
    print("\n" + "=" * 80)
    print("TEST 2: Search for enzyme engineering and thermostability")
    print("=" * 80)
    results2 = server.search_papers("enzyme engineering thermostability", max_results=3)
    
    if isinstance(results2, list):
        print(f"\nFound {len(results2)} papers:\n")
        for i, paper in enumerate(results2, 1):
            print(f"{i}. {paper['title']}")
            print(f"   Published: {paper['published']}")
            print()