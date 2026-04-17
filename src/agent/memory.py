from datetime import datetime
from typing import List, Optional


MAX_SESSIONS = 10


def save_session(query: str, state: dict) -> None:
    """Save a research session to st.session_state history (max 10)."""
    import streamlit as st

    if "research_history" not in st.session_state:
        st.session_state["research_history"] = []

    session = {
        "session_id": state.get("session_id", ""),
        "query": query,
        "report_markdown": state.get("report_markdown", ""),
        "qa_history": state.get("qa_history", []),
        "quality_score": state.get("quality_score", 0.0),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "source_count": len(state.get("search_results", [])),
    }

    history = st.session_state["research_history"]
    history.insert(0, session)
    st.session_state["research_history"] = history[:MAX_SESSIONS]


def get_history() -> List[dict]:
    """Return list of past research sessions."""
    import streamlit as st
    return st.session_state.get("research_history", [])


def get_session(session_id: str) -> Optional[dict]:
    """Retrieve a specific past session by ID."""
    for s in get_history():
        if s["session_id"] == session_id:
            return s
    return None
