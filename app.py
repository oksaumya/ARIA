import streamlit as st

st.set_page_config(
    page_title="ARIA - Research AI",
    page_icon="🔬",
    layout="wide",
)

st.title("ARIA")
st.markdown("### Autonomous Research Intelligence Assistant")
st.markdown("An intelligent two-phase research system, from classical NLP to autonomous agentic AI.")
st.markdown("---")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("#### 1: NLP Analysis")
    st.markdown(
        """
        Traditional ML/NLP pipeline for document analysis:
        - **Extractive Summarization** via TextRank
        - **Keyword Extraction** via TF-IDF with visualizations
        - **Topic Modeling** via LDA
        - Upload PDF / DOCX / TXT or search Wikipedia
        """
    )
    st.page_link("pages/1_NLP_Analysis.py", label="Open NLP Analysis", icon="📄")

with col2:
    st.markdown("#### 2: ARIA Agent")
    st.markdown(
        """
        Agentic AI research pipeline powered by LangGraph + Groq:
        - **Autonomous multi-query web search** via DuckDuckGo
        - **Iterative quality control** : retries until coverage is sufficient
        - **Structured reports**: Abstract · Key Findings · Conclusion · Sources
        - **Follow-up Q&A** grounded in the report
        - **PDF export** with full Q&A history
        """
    )
    st.page_link("pages/2_AI_Research_Agent.py", label="Open ARIA Agent", icon="🤖")

st.markdown("---")
st.caption("Built with Streamlit · LangGraph · Groq (Llama 3.3 70B) · DuckDuckGo Search")
