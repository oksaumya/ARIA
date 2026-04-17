from typing import TypedDict, List, Optional, Annotated
import operator


class SearchResult(TypedDict):
    url: str
    title: str
    snippet: str
    full_content: Optional[str]
    source_quality: float


class ResearchReport(TypedDict):
    title: str
    abstract: str
    key_findings: List[str]
    conclusion: str
    sources: List[dict]


class ResearchState(TypedDict):
    query: str
    search_queries: List[str]
    search_results: List[SearchResult]
    failed_urls: List[str]
    source_summaries: List[str]
    aggregated_text: str
    quality_score: float
    quality_feedback: str
    missing_aspects: List[str]
    retry_count: int
    max_retries: int
    report: Optional[ResearchReport]
    report_markdown: str
    qa_history: List[dict]
    session_id: str
    research_timestamp: str
    errors: Annotated[List[str], operator.add]
    status: str
