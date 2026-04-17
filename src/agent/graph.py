from langgraph.graph import StateGraph, END

from src.agent.state import ResearchState
from src.agent.nodes import (
    plan_research,
    web_searcher,
    content_fetcher,
    source_summarizer,
    aggregator,
    quality_validator,
    report_generator,
)


def should_retry(state: ResearchState) -> str:
    """Decide whether to retry with more searches or generate the final report."""
    score = state.get("quality_score", 1.0)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if score < 0.6 and retry_count < max_retries:
        return "retry"
    return "generate"


def increment_retry(state: ResearchState) -> dict:
    """Increment retry counter before re-entering the search loop."""
    return {"retry_count": state.get("retry_count", 0) + 1}


def build_graph() -> StateGraph:
    builder = StateGraph(ResearchState)

    builder.add_node("plan_research", plan_research)
    builder.add_node("web_searcher", web_searcher)
    builder.add_node("content_fetcher", content_fetcher)
    builder.add_node("source_summarizer", source_summarizer)
    builder.add_node("aggregator", aggregator)
    builder.add_node("quality_validator", quality_validator)
    builder.add_node("increment_retry", increment_retry)
    builder.add_node("report_generator", report_generator)

    builder.set_entry_point("plan_research")
    builder.add_edge("plan_research", "web_searcher")
    builder.add_edge("web_searcher", "content_fetcher")
    builder.add_edge("content_fetcher", "source_summarizer")
    builder.add_edge("source_summarizer", "aggregator")
    builder.add_edge("aggregator", "quality_validator")
    builder.add_conditional_edges(
        "quality_validator",
        should_retry,
        {
            "retry": "increment_retry",
            "generate": "report_generator",
        },
    )
    builder.add_edge("increment_retry", "web_searcher")
    builder.add_edge("report_generator", END)

    return builder.compile()


# Module-level compiled graph instance
research_graph = build_graph()
