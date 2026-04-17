import time
from typing import List
from src.agent.state import SearchResult


class DuckDuckGoSearcher:
    def __init__(self, max_results_per_query: int = 5, sleep_between: float = 1.0):
        self.max_results = max_results_per_query
        self.sleep_between = sleep_between

    def search(self, query: str) -> List[SearchResult]:
        """Search DuckDuckGo with retry logic."""
        from duckduckgo_search import DDGS

        for attempt in range(3):
            try:
                results = []
                with DDGS() as ddgs:
                    for r in ddgs.text(query, max_results=self.max_results):
                        results.append(SearchResult(
                            url=r.get("href", ""),
                            title=r.get("title", ""),
                            snippet=r.get("body", "")[:500],
                            full_content=None,
                            source_quality=0.5,
                        ))
                return results
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    return []

    def batch_search(self, queries: List[str]) -> List[SearchResult]:
        """Search all queries, deduplicate by URL, cap at 15 results."""
        seen_urls = set()
        all_results = []

        for query in queries:
            results = self.search(query)
            for r in results:
                if r["url"] and r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    all_results.append(r)
                    if len(all_results) >= 15:
                        return all_results
            time.sleep(self.sleep_between)

        return all_results
