import uuid
import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="ARIA — AI Research Agent",
    page_icon="🔬",
    layout="wide",
)


_import_errors = []
try:
    from src.agent.graph import research_graph
    from src.agent.qa_agent import answer_question
    from src.agent.report import export_pdf
    from src.agent.memory import save_session, get_history
except ImportError as e:
    _import_errors.append(str(e))


#Sidebar
st.sidebar.title("ARIA Agent")
st.sidebar.caption("Autonomous Research Intelligence Assistant")
st.sidebar.markdown("---")

# API Key input — auto-load from secrets if available
_auto_key = ""
try:
    _auto_key = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    pass

api_key_input = st.sidebar.text_input(
    "Groq API Key",
    value=_auto_key,
    type="password",
    placeholder="gsk_...",
    help="Free key at console.groq.com",
)
if api_key_input:
    st.session_state["GROQ_API_KEY"] = api_key_input

api_key = st.session_state.get("GROQ_API_KEY", _auto_key)

# Settings
st.sidebar.markdown("**Settings**")
max_retries = st.sidebar.slider(
    "Max Research Iterations", min_value=1, max_value=3, value=2,
    help="Higher = more thorough but slower"
)
show_sources = st.sidebar.checkbox("Show source details", value=True)

st.sidebar.markdown("---")

# Research history
st.sidebar.markdown("**Research History**")
history = get_history() if not _import_errors else []

if history:
    for i, session in enumerate(history):
        label = session["query"][:28] + "..." if len(session["query"]) > 28 else session["query"]
        with st.sidebar.expander(f"{session['timestamp']} — {label}", expanded=False):
            st.caption(f"Quality: {session['quality_score']:.0%} · Sources: {session['source_count']}")
            if st.button("Load report", key=f"load_{i}"):
                st.session_state["loaded_report"] = session["report_markdown"]
                st.session_state["loaded_qa"] = session["qa_history"]
                st.session_state["current_query"] = session["query"]
                st.rerun()
else:
    st.sidebar.caption("No research history yet.")


# Main content
st.title("ARIA: Autonomous Research Intelligence Assistant")
st.markdown(
    "Powered by **LangGraph** · **Groq (Llama 3.3 70B)** · **DuckDuckGo**  \n"
    "Enter any research question and ARIA will search the web, synthesize findings across sources, "
    "and produce a structured report with follow-up Q&A and PDF export."
)

if _import_errors:
    st.error(
        f"Missing dependencies. Run:\n```\npip install -r requirements.txt\n```\n\nDetails: {_import_errors[0]}"
    )
    st.stop()

# Loaded session from history 
if "loaded_report" in st.session_state:
    st.info(f"Loaded report: **{st.session_state.get('current_query', '')}**")
    st.markdown(st.session_state["loaded_report"])

    loaded_qa = st.session_state.get("loaded_qa", [])
    if loaded_qa:
        st.subheader("Previous Q&A")
        for qa in loaded_qa:
            with st.chat_message("user"):
                st.write(qa["question"])
            with st.chat_message("assistant"):
                st.write(qa["answer"])

    try:
        pdf_bytes = export_pdf(st.session_state["loaded_report"], loaded_qa)
        st.download_button(
            "Download Report as PDF",
            data=pdf_bytes,
            file_name="aria_research_report.pdf",
            mime="application/pdf",
        )
    except Exception as e:
        st.caption(f"PDF export unavailable: {e}")

    if st.button("Start new research"):
        del st.session_state["loaded_report"]
        st.rerun()
    st.stop()


# Two-column layout 
left_col, right_col = st.columns([6, 4])

with left_col:
    query = st.text_input(
        "Research Query",
        placeholder="e.g. What are the latest advances in CRISPR gene editing?",
        key="research_query",
    )

    run_btn = st.button(
        "Run ARIA Agent",
        type="primary",
        disabled=not api_key or not query,
    )

    if not api_key:
        st.warning("Add your Groq API key in the sidebar to get started. Free at [console.groq.com](https://console.groq.com).")

    # Agent execution
    if run_btn and query and api_key:
        st.session_state["agent_state"] = None
        st.session_state["report_ready"] = False
        st.session_state["qa_history"] = []

        initial_state = {
            "query": query,
            "search_queries": [],
            "search_results": [],
            "failed_urls": [],
            "source_summaries": [],
            "aggregated_text": "",
            "quality_score": 0.0,
            "quality_feedback": "",
            "missing_aspects": [],
            "retry_count": 0,
            "max_retries": max_retries,
            "report": None,
            "report_markdown": "",
            "qa_history": [],
            "session_id": str(uuid.uuid4()),
            "research_timestamp": datetime.now().isoformat(),
            "errors": [],
            "status": "Starting...",
            "api_key": api_key,
        }

        node_labels = {
            "plan_research":      "Planning research strategy...",
            "web_searcher":       "Searching the web with DuckDuckGo...",
            "content_fetcher":    "Fetching full source content...",
            "source_summarizer":  "Summarizing each source with Llama 3.3...",
            "aggregator":         "Aggregating findings into synthesis...",
            "quality_validator":  "Validating research quality...",
            "increment_retry":    "Refining search for better coverage...",
            "report_generator":   "Generating structured research report...",
        }

        accumulated_state = dict(initial_state)
        accumulated_state["errors"] = []

        with st.status("ARIA is researching...", expanded=True) as status_box:
            try:
                for event in research_graph.stream(initial_state):
                    node_name = list(event.keys())[0]
                    node_output = event[node_name]

                    for k, v in node_output.items():
                        if k == "errors" and isinstance(v, list):
                            accumulated_state["errors"] = accumulated_state.get("errors", []) + v
                        else:
                            accumulated_state[k] = v

                    label = node_labels.get(node_name, f"Running {node_name}...")

                    if node_name == "quality_validator":
                        score = node_output.get("quality_score", 0)
                        verdict = "Good coverage!" if score >= 0.6 else "Searching for more..."
                        label = f"Quality score: {score:.0%} — {verdict}"

                    status_box.write(f"✓ {label}")

                status_box.update(label="Research complete!", state="complete")

            except Exception as e:
                status_box.update(label=f"Error: {str(e)}", state="error")
                st.error(f"ARIA encountered an error: {str(e)}")
                st.stop()

        st.session_state["agent_state"] = accumulated_state
        st.session_state["report_ready"] = True
        st.session_state["qa_history"] = accumulated_state.get("qa_history", [])
        save_session(query, accumulated_state)

    # Report display
    if st.session_state.get("report_ready"):
        state = st.session_state["agent_state"]
        report_md = state.get("report_markdown", "")

        st.markdown("---")
        st.markdown(report_md)

        q_score = state.get("quality_score", 0)
        retries = state.get("retry_count", 0)
        errors = state.get("errors", [])

        meta_cols = st.columns(3)
        meta_cols[0].metric("Quality Score", f"{q_score:.0%}")
        meta_cols[1].metric("Sources Found", len(state.get("search_results", [])))
        meta_cols[2].metric("Research Iterations", retries + 1)

        if errors:
            with st.expander("Warnings during research"):
                for err in errors:
                    st.caption(f"• {err}")

        if show_sources:
            with st.expander("Source Details"):
                for r in state.get("search_results", []):
                    st.markdown(f"**[{r['title']}]({r['url']})**")
                    st.caption(r.get("snippet", ""))
                    st.markdown("---")

        qa_hist = st.session_state.get("qa_history", [])
        try:
            pdf_bytes = export_pdf(report_md, qa_hist)
            st.download_button(
                "Download Report as PDF",
                data=pdf_bytes,
                file_name=f"aria_{query[:30].replace(' ', '_')}.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.caption(f"PDF export unavailable: {e}")


# Right column: Q&A chat
with right_col:
    st.subheader("Ask ARIA")

    if not st.session_state.get("report_ready"):
        st.info("Run a research query to unlock Q&A. Ask follow-up questions grounded in the generated report.")
    else:
        st.caption("Questions are answered using only the research report above — no hallucination beyond sources.")

        qa_history = st.session_state.get("qa_history", [])
        for qa in qa_history:
            with st.chat_message("user"):
                st.write(qa["question"])
            with st.chat_message("assistant"):
                st.write(qa["answer"])

        if follow_up := st.chat_input("Ask a follow-up question..."):
            state = st.session_state["agent_state"]

            with st.chat_message("user"):
                st.write(follow_up)

            with st.chat_message("assistant"):
                with st.spinner("ARIA is thinking..."):
                    answer, updated_state = answer_question(state, follow_up, api_key=api_key)
                st.write(answer)

            st.session_state["agent_state"] = updated_state
            st.session_state["qa_history"] = updated_state.get("qa_history", [])
            st.rerun()
