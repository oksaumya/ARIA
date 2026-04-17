import time
from typing import List, Tuple
from urllib.parse import urlparse
from src.agent.state import SearchResult

# Domains that are considered high-quality academic / institutional sources
_HIGH_TRUST_DOMAINS = {
    "wikipedia.org", "arxiv.org", "pubmed.ncbi.nlm.nih.gov", "scholar.google.com",
    "nature.com", "sciencedirect.com", "springer.com", "ieee.org", "acm.org",
    "mit.edu", "stanford.edu", "harvard.edu", "ox.ac.uk", "cam.ac.uk",
    "researchgate.net", "semanticscholar.org", "jstor.org",
}
_MEDIUM_TRUST_TLDS = {".edu", ".gov", ".ac.uk", ".ac.in", ".org"}


def _score_url(url: str) -> float:
    """Heuristic source-quality score in [0.0, 1.0] based on URL signals.

    Scoring breakdown:
    - Base score: 0.4
    - HTTPS:      +0.05
    - High-trust domain (academic/gov/org):  +0.30
    - Medium-trust TLD (.edu/.gov/.org etc): +0.15 (if not already high-trust)
    - Shallow URL path (≤ 3 segments):      +0.10  (avoids deep paginated junk)
    """
    if not url:
        return 0.4

    try:
        parsed = urlparse(url)
    except Exception:
        return 0.4

    score = 0.4

    # HTTPS bonus
    if parsed.scheme == "https":
        score += 0.05

    hostname = parsed.hostname or ""
    # Strip leading 'www.'
    hostname = hostname.removeprefix("www.")

    # High-trust domain check
    is_high_trust = any(hostname == d or hostname.endswith("." + d) for d in _HIGH_TRUST_DOMAINS)
    if is_high_trust:
        score += 0.30
    else:
        # Medium-trust TLD check
        if any(hostname.endswith(tld) for tld in _MEDIUM_TRUST_TLDS):
            score += 0.15

    # Shallow path bonus — fewer path segments = cleaner, more authoritative page
    path_depth = len([s for s in parsed.path.split("/") if s])
    if path_depth <= 3:
        score += 0.10

    return round(min(score, 1.0), 2)


class DuckDuckGoSearcher:
    def __init__(self, max_results_per_query: int = 5, sleep_between: float = 1.0):
        self.max_results = max_results_per_query
        self.sleep_between = sleep_between

    def search(self, query: str) -> Tuple[List[SearchResult], List[str]]:
        """Search DuckDuckGo with retry logic."""
        errors: List[str] = []
        try:
            from ddgs import DDGS
        except ImportError as e:
            return [], [f"DuckDuckGo search dependency is missing. Run: pip install ddgs. Error: {e}"]

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
                        raw_url = r.get("href", "")
                        results.append(SearchResult(
                            url=raw_url,
                            title=r.get("title", ""),
                            snippet=r.get("body", "")[:500],
                            full_content=None,
                            source_quality=_score_url(raw_url),
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
                    source_quality=_score_url(page.url),
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
