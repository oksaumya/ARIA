import time
from typing import List, Tuple
from src.agent.state import SearchResult


class DuckDuckGoSearcher:
    def __init__(self, max_results_per_query: int = 5, sleep_between: float = 1.0):
        self.max_results = max_results_per_query
        self.sleep_between = sleep_between

    def search(self, query: str) -> Tuple[List[SearchResult], List[str]]:
        """Search DuckDuckGo with retry logic."""
        errors: List[str] = []
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS
            except ImportError as e:
                return [], [f"DuckDuckGo search dependency is missing: {e}"]

        query = query.strip()
        if not query:
            return [], ["DuckDuckGo search skipped because the query was empty."]

        search_attempts = (
            {"region": "us-en", "safesearch": "off", "backend": "html"},
            {"region": "us-en", "safesearch": "off", "backend": "api"},
        )

        for attempt in range(3):
            try:
                with DDGS() as ddgs:
                    search_kwargs = search_attempts[min(attempt, len(search_attempts) - 1)]
                    results = []
                    for r in ddgs.text(
                        query,
                        max_results=self.max_results,
                        timelimit=None,
                        **search_kwargs,
                    ):
                        results.append(SearchResult(
                            url=r.get("href", ""),
                            title=r.get("title", ""),
                            snippet=r.get("body", "")[:500],
                            full_content=None,
                            source_quality=0.5,
                        ))
                if not results:
                    errors.append(f"DuckDuckGo returned no results for query: {query}")
                return results, errors
            except Exception as e:
                errors.append(f"DuckDuckGo search attempt {attempt + 1} failed for '{query}': {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    return [], errors

    def batch_search(self, queries: List[str]) -> Tuple[List[SearchResult], List[str]]:
        """Search all queries, deduplicate by URL, cap at 15 results."""
        seen_urls = set()
        all_results = []
        all_errors: List[str] = []

        for query in queries:
            results, errors = self.search(query)
            all_errors.extend(errors)
            for r in results:
                if r["url"] and r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    all_results.append(r)
                    if len(all_results) >= 15:
                        return all_results, all_errors
            time.sleep(self.sleep_between)

        return all_results, all_errors


class WikipediaSearcher:
    def __init__(self, max_results_per_query: int = 3, sleep_between: float = 0.5):
        self.max_results = max_results_per_query
        self.sleep_between = sleep_between

    def search(self, query: str) -> Tuple[List[SearchResult], List[str]]:
        """Search Wikipedia for relevant pages and summaries."""
        try:
            import wikipedia
        except ImportError:
            return [], ["Wikipedia dependency is missing."]

        query = query.strip()
        if not query:
            return [], ["Wikipedia search skipped because the query was empty."]

        try:
            wikipedia.set_lang("en")
            titles = wikipedia.search(query, results=self.max_results)
        except Exception as e:
            return [], [f"Wikipedia search failed for '{query}': {e}"]

        results: List[SearchResult] = []
        errors: List[str] = []
        for title in titles:
            try:
                page = wikipedia.page(title, auto_suggest=False, redirect=True)
                summary = wikipedia.summary(page.title, sentences=2, auto_suggest=False, redirect=True)
                results.append(SearchResult(
                    url=page.url,
                    title=page.title,
                    snippet=summary[:500],
                    full_content=summary,
                    source_quality=0.8,
                ))
            except Exception as e:
                errors.append(f"Wikipedia page fetch failed for '{title}': {e}")
                continue

        if not results:
            errors.append(f"Wikipedia returned no results for query: {query}")

        return results, errors

    def batch_search(self, queries: List[str]) -> Tuple[List[SearchResult], List[str]]:
        """Search all queries and deduplicate by URL."""
        seen_urls = set()
        all_results = []
        all_errors: List[str] = []

        for query in queries:
            results, errors = self.search(query)
            all_errors.extend(errors)
            for r in results:
                if r["url"] and r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    all_results.append(r)
            time.sleep(self.sleep_between)

        return all_results, all_errors
