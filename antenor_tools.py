# ==============================================================================
# Project Prometheus: Antenor Tools
# Version 0.1
#
# Agent: PrometheusAI
# Mission: Provide the kernel with the tools necessary for autonomous learning,
#          starting with web browsing.
#
# Description:
# This module provides functions for searching the web and scraping content
# from URLs. It is designed to be used by the Prometheus Kernel.
# ==============================================================================

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

# We use duckduckgo-search as it doesn't require an API key.

def search_web(query: str, num_results: int = 3) -> list[dict]:
    """
    Performs a web search using DuckDuckGo and returns the results.

    Args:
        query: The search term.
        num_results: The number of results to return.

    Returns:
        A list of dictionaries, where each dictionary contains 'title', 'href', and 'body'.
    """
    print(f"[Antenor] Performing web search for: '{query}'")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
        print(f"[Antenor] Found {len(results)} results.")
        return results
    except Exception as e:
        print(f"[Antenor] Error during web search: {e}")
        return []

def scrape_url(url: str) -> str:
    """
    Scrapes the primary text content from a given URL.

    Args:
        url: The URL to scrape.

    Returns:
        The cleaned text content of the page, or an empty string if scraping fails.
    """
    print(f"[Antenor] Scraping content from: {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()

        # Get text, strip leading/trailing whitespace, and join lines
        text = ' '.join(t.strip() for t in soup.stripped_strings)
        
        print(f"[Antenor] Successfully scraped {len(text)} characters.")
        return text
    except Exception as e:
        print(f"[Antenor] Error scraping URL {url}: {e}")
        return ""

# --- Standalone Test ---
if __name__ == "__main__":
    print("Performing standalone test of Antenor tools...")
    
    # Test Search
    search_results = search_web("What is the Kármán line?")
    assert len(search_results) > 0
    print("Search results:")
    for res in search_results:
        print(f"  - {res['title']}: {res['href']}")

    # Test Scrape
    if search_results:
        first_url = search_results[0]['href']
        scraped_content = scrape_url(first_url)
        assert len(scraped_content) > 100
        print(f"\nScraped content preview (first 200 chars):\n{scraped_content[:200]}...")

    print("\nAntenor tools standalone tests complete.")
