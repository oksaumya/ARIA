import streamlit as st

st.set_page_config(
    page_title="ARIA - Research AI",
    page_icon="🔬",
    layout="wide",
)

# Style page links to look like buttons
st.markdown("""
<style>
[data-testid="stPageLink"] a {
    display: block;
    background-color: #FF4B4B;
    color: white !important;
    padding: 0.6rem 1.2rem;
    border-radius: 0.5rem;
    text-decoration: none !important;
    font-weight: 600;
    font-size: 1rem;
    text-align: center;
    transition: background-color 0.2s ease;
}
[data-testid="stPageLink"] a:hover {
    background-color: #cc3a3a;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.title("ARIA")
st.markdown("### Autonomous Research Intelligence Assistant")
st.markdown("An intelligent two-phase research system - from classical NLP to autonomous agentic AI.")
st.markdown("---")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("#### NLP Analysis")
    st.markdown(
        """
        Traditional ML/NLP pipeline for document analysis:
        - **Extractive Summarization** via TextRank
        - **Keyword Extraction** via TF-IDF with visualizations
        - **Topic Modeling** via LDA
        - Upload PDF / DOCX / TXT or search Wikipedia
        """
    )
    st.page_link("pages/1_NLP_Analysis.py", label="NLP Analysis")

with col2:
    st.markdown("#### ARIA Agent")
    st.markdown(
        """
        Agentic AI research pipeline powered by LangGraph + Groq:
        - **Autonomous multi-query web search** via DuckDuckGo
        - **Iterative quality control** - retries until coverage is sufficient
        - **Structured reports**: Abstract, Key Findings, Conclusion, Sources
        - **Follow-up Q&A** grounded in the report
        - **PDF export** with full Q&A history
        """
    )
    st.page_link("pages/2_AI_Research_Agent.py", label="ARIA Agent")

st.markdown("---")
st.caption("Built with Streamlit · LangGraph · Groq (Llama 3.3 70B) · DuckDuckGo Search")
