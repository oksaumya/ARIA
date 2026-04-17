import json
import time
import urllib3
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage

from src.agent.state import ResearchState, SearchResult
from src.agent.web_search import DuckDuckGoSearcher

# Disable SSL warnings for web fetching
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
http = urllib3.PoolManager(timeout=urllib3.Timeout(connect=3.0, read=5.0))


def _get_llm(state: ResearchState):
    from src.agent.llm import get_llm
    api_key = state.get("api_key", None)
    return get_llm(api_key=api_key)


def plan_research(state: ResearchState) -> dict:
    """Decompose query into 3–5 targeted sub-queries using Grok."""
    query = state["query"]
    llm = _get_llm(state)

    system = (
        "You are a research planning assistant. Given a research topic, generate 3 to 5 "
        "distinct search queries that cover different facets: overview, recent developments, "
        "applications, criticisms or limitations, and future directions. "
        'Return ONLY a JSON object with key "queries" containing a list of strings. '
        "Example: {\"queries\": [\"query 1\", \"query 2\", \"query 3\"]}"
    )

    try:
        response = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=f"Research topic: {query}"),
        ])
        data = json.loads(response.content)
        queries = data.get("queries", [query])
        if not isinstance(queries, list) or not queries:
            queries = [query]
    except Exception as e:
        queries = [query]

    return {
        "search_queries": queries,
        "status": "Planning complete — generated search queries.",
        "errors": [],
    }


def web_searcher(state: ResearchState) -> dict:
    """Search DuckDuckGo for all sub-queries."""
    queries = state.get("search_queries", [state["query"]])
    missing = state.get("missing_aspects", [])
    if missing:
        queries = queries + missing

    searcher = DuckDuckGoSearcher(max_results_per_query=5, sleep_between=1.0)
    existing_urls = {r["url"] for r in state.get("search_results", [])}

    new_results = []
    for q in queries:
        results = searcher.search(q)
        for r in results:
            if r["url"] not in existing_urls:
                existing_urls.add(r["url"])
                new_results.append(r)
        time.sleep(1.0)

    all_results = state.get("search_results", []) + new_results
    all_results = all_results[:15]

    return {
        "search_results": all_results,
        "status": f"Web search complete — found {len(all_results)} sources.",
        "errors": [],
    }


def content_fetcher(state: ResearchState) -> dict:
    """Fetch and extract text content from each search result URL."""
    from bs4 import BeautifulSoup

    results = state.get("search_results", [])
    failed_urls = list(state.get("failed_urls", []))
    updated = []

    for r in results:
        if r.get("full_content") is not None:
            updated.append(r)
            continue

        url = r["url"]
        if not url or url in failed_urls:
            updated.append({**r, "full_content": r["snippet"]})
            continue

        try:
            resp = http.request(
                "GET", url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"},
                redirect=True,
            )
            if resp.status == 200:
                soup = BeautifulSoup(resp.data, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                paragraphs = soup.find_all("p")
                text = " ".join(p.get_text(strip=True) for p in paragraphs)
                text = text[:3000] if text else r["snippet"]
                updated.append({**r, "full_content": text})
            else:
                failed_urls.append(url)
                updated.append({**r, "full_content": r["snippet"]})
        except Exception:
            failed_urls.append(url)
            updated.append({**r, "full_content": r["snippet"]})

    return {
        "search_results": updated,
        "failed_urls": failed_urls,
        "status": f"Content fetched — {len(failed_urls)} URLs failed.",
        "errors": [],
    }


def source_summarizer(state: ResearchState) -> dict:
    """Summarize each source with Grok, sequentially to respect rate limits."""
    query = state["query"]
    results = state.get("search_results", [])
    llm = _get_llm(state)

    summaries = []
    system = (
        "You are a research assistant. Summarize the following source content in 3–5 sentences, "
        "focusing specifically on how it relates to the research query. "
        "If the content is irrelevant to the query, respond with exactly: IRRELEVANT"
    )

    for r in results:
        content = r.get("full_content") or r.get("snippet", "")
        if not content:
            continue
        try:
            resp = llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=f"Query: {query}\n\nSource title: {r['title']}\n\nContent:\n{content}"),
            ])
            summary = resp.content.strip()
            if summary != "IRRELEVANT":
                summaries.append(f"[{r['title']}]({r['url']}): {summary}")
            time.sleep(1.0)
        except Exception as e:
            summaries.append(f"[{r['title']}]: {r['snippet']}")
            time.sleep(1.0)

    return {
        "source_summaries": summaries,
        "status": f"Summarized {len(summaries)} relevant sources.",
        "errors": [],
    }


def aggregator(state: ResearchState) -> dict:
    """Aggregate all source summaries into a unified synthesis."""
    query = state["query"]
    summaries = state.get("source_summaries", [])
    llm = _get_llm(state)

    if not summaries:
        return {
            "aggregated_text": "No relevant sources found.",
            "status": "Aggregation complete.",
            "errors": ["No source summaries to aggregate."],
        }

    combined = "\n\n".join(summaries)
    system = (
        "You are a research synthesis expert. Given the following per-source summaries, "
        "write a unified synthesis that: identifies consensus findings, notes contradictions, "
        "highlights knowledge gaps, and directly addresses the research query. "
        "Write 3–5 well-developed paragraphs. Be specific and evidence-based."
    )

    try:
        resp = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=f"Research query: {query}\n\nSource summaries:\n{combined}"),
        ])
        aggregated = resp.content.strip()
    except Exception as e:
        aggregated = combined

    return {
        "aggregated_text": aggregated,
        "status": "Aggregation complete.",
        "errors": [],
    }


def quality_validator(state: ResearchState) -> dict:
    """Score the research quality and identify gaps for potential retry."""
    query = state["query"]
    aggregated = state.get("aggregated_text", "")
    llm = _get_llm(state)

    system = (
        "You are a research quality evaluator. Score the following research synthesis on three dimensions:\n"
        "- Coverage (0.0–0.4): Does it address all major aspects of the query?\n"
        "- Evidence (0.0–0.3): Are claims supported by source references?\n"
        "- Coherence (0.0–0.3): Is the narrative well-organized and logical?\n\n"
        "Return ONLY a JSON object with keys: "
        '"score" (float, sum of dimensions), "feedback" (string), "missing_aspects" (list of strings). '
        'Example: {"score": 0.75, "feedback": "Good coverage but lacks recent data.", "missing_aspects": ["recent 2024 developments"]}'
    )

    try:
        resp = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=f"Research query: {query}\n\nSynthesis:\n{aggregated}"),
        ])
        data = json.loads(resp.content)
        score = float(data.get("score", 0.5))
        feedback = data.get("feedback", "")
        missing = data.get("missing_aspects", [])
    except Exception as e:
        score = 0.7
        feedback = "Quality check could not be completed."
        missing = []

    return {
        "quality_score": min(max(score, 0.0), 1.0),
        "quality_feedback": feedback,
        "missing_aspects": missing,
        "status": f"Quality score: {score:.2f}. {feedback}",
        "errors": [],
    }


def report_generator(state: ResearchState) -> dict:
    """Generate the final structured research report."""
    query = state["query"]
    aggregated = state.get("aggregated_text", "")
    results = state.get("search_results", [])
    llm = _get_llm(state)

    sources_list = [
        {"title": r["title"], "url": r["url"]}
        for r in results if r.get("url")
    ][:10]

    system = (
        "You are an expert research report writer. Generate a comprehensive, structured research report "
        "based on the provided synthesis. Return ONLY a JSON object with these exact keys:\n"
        '- "title": string (descriptive report title)\n'
        '- "abstract": string (2–3 sentence overview)\n'
        '- "key_findings": list of strings (5–8 specific, evidence-based bullet points)\n'
        '- "conclusion": string (1 well-developed paragraph)\n'
        '- "sources": list of {"title": string, "url": string} objects\n\n'
        "Base the report strictly on the provided synthesis. Do not add information not present in the synthesis."
    )

    try:
        resp = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=(
                f"Research query: {query}\n\n"
                f"Research synthesis:\n{aggregated}\n\n"
                f"Available sources: {json.dumps(sources_list)}"
            )),
        ])

        raw = resp.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data = json.loads(raw)
        report = {
            "title": data.get("title", f"Research Report: {query}"),
            "abstract": data.get("abstract", ""),
            "key_findings": data.get("key_findings", []),
            "conclusion": data.get("conclusion", ""),
            "sources": data.get("sources", sources_list),
        }
    except Exception as e:
        report = {
            "title": f"Research Report: {query}",
            "abstract": aggregated[:300] + "..." if len(aggregated) > 300 else aggregated,
            "key_findings": [aggregated[i:i+200] for i in range(0, min(len(aggregated), 1000), 200)],
            "conclusion": "Report generation encountered an error. The synthesis above represents the research findings.",
            "sources": sources_list,
        }

    markdown = _render_report_markdown(report)

    return {
        "report": report,
        "report_markdown": markdown,
        "status": "Research report generated successfully.",
        "errors": [],
    }


def _render_report_markdown(report: dict) -> str:
    """Convert a report dict to formatted markdown."""
    lines = [
        f"# {report['title']}",
        "",
        "## Abstract",
        report.get("abstract", ""),
        "",
        "## Key Findings",
    ]
    for finding in report.get("key_findings", []):
        lines.append(f"- {finding}")
    lines += [
        "",
        "## Conclusion",
        report.get("conclusion", ""),
        "",
        "## Sources",
    ]
    for s in report.get("sources", []):
        title = s.get("title", "Unknown")
        url = s.get("url", "#")
        lines.append(f"- [{title}]({url})")

    return "\n".join(lines)
