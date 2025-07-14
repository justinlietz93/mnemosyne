import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from antenor_tools import search_web, scrape_url

# Test search_web
query = "Recent quantum computing advances 2025"
results = search_web(query, num_results=3)
print("Search Results:")
for res in results:
    print(f"- Title: {res['title']}")
    print(f"  URL: {res['href']}")
    print(f"  Snippet: {res['body'][:100]}...")  # Preview

if results:
    # Test scrape_url with first result
    first_url = results[0]['href']
    content = scrape_url(first_url)
    print("\nScraped Content Preview:")
    print(content[:500] + "..." if content else "No content scraped.")

print("Test completed. Check for any errors above.")