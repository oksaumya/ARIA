# ARIA - Autonomous Research Intelligence Assistant

> An intelligent two-phase research system: from classical NLP to a fully autonomous AI research agent.

---

## Overview

ARIA is built as a two-milestone project that demonstrates the progression from traditional machine learning and NLP to modern agentic AI workflows.

| Milestone | Approach | Status |
|:---|:---|:---|
| **M1 — NLP Analysis** | Classical ML: TF-IDF, LDA, TextRank | Complete |
| **M2 — ARIA Agent** | LangGraph + Groq (Llama 3.3 70B) + DuckDuckGo | Complete |

---

## Quick Start

### 1. Clone & Setup (one command)

```bash
git clone https://github.com/oksaumya/ARIA.git
cd ARIA
bash setup.sh
```

`setup.sh` creates an isolated `.venv` virtual environment, installs all dependencies, and downloads the required spaCy and NLTK models automatically.

### 2. Add your Groq API Key

Open `.streamlit/secrets.toml` and add your key:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

Get a **free** key at [console.groq.com](https://console.groq.com) — no credit card required.

### 3. Run the App

```bash
source .venv/bin/activate
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Manual Setup (alternative to `setup.sh`)

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download language models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

# Run
streamlit run app.py
```

---

## Project Structure

```
ARIA/
  app.py                          # Home page (Streamlit multi-page entry)
  setup.sh                        # One-shot environment setup script
  requirements.txt                # All Python dependencies
  pages/
    1_NLP_Analysis.py             # Milestone 1 — NLP pipeline UI
    2_AI_Research_Agent.py        # Milestone 2 — ARIA agent UI
  src/
    data_fetcher.py               # PDF / DOCX / TXT / Wikipedia extraction
    nlp_pipeline.py               # TF-IDF, LDA, TextRank
    agent/
      state.py                    # LangGraph ResearchState TypedDict
      graph.py                    # Graph topology + conditional retry edges
      nodes.py                    # 7 node functions (plan → search → fetch → summarize → aggregate → validate → report)
      llm.py                      # Groq API client (Llama 3.3 70B)
      web_search.py               # DuckDuckGo wrapper with retry
      report.py                   # Markdown renderer + fpdf2 PDF export
      memory.py                   # Session history (st.session_state)
      qa_agent.py                 # Follow-up Q&A grounded in report
  .streamlit/
    secrets.toml                  # API keys (gitignored — never commit)
```

---

## Milestone 1 - NLP Analysis System

**Objective:** Analyze research documents using classical NLP without any LLMs.

**Features:**
- Upload PDF, DOCX, or TXT files — or search Wikipedia directly
- **Extractive Summary** via TextRank (Sumy)
- **Keyword Extraction** via TF-IDF with bar chart visualization
- **Topic Modeling** via LDA (4 topics, configurable)
- Raw text preview

**Tech:** spaCy · scikit-learn · Sumy · Gensim · PyMuPDF

---

## Milestone 2 - ARIA Agent

**Objective:** Autonomously research any topic by searching the web, synthesizing multiple sources, and generating a structured report.

**LangGraph Workflow:**

```
plan_research → web_searcher → content_fetcher → source_summarizer
             → aggregator → quality_validator
                                   |
                    +--------------+---------------+
                    |                              |
              score < 0.6                   score >= 0.6
              retry_count < max             (or max reached)
                    |                              |
             web_searcher (retry)          report_generator → END
```

**Features:**
- **Query planning** :- Grok decomposes query into 3–5 targeted sub-questions
- **Web search** :- DuckDuckGo (free, no API key needed for search)
- **Content fetching** :- Fetches and parses full page text via BeautifulSoup
- **Per-source AI summaries** :- Groq summarizes each source in context
- **Iterative quality control** :- Re-searches if coverage score < 60%
- **Structured report** :- Title · Abstract · Key Findings · Conclusion · Sources
- **Follow-up Q&A** :- Chat interface grounded in the report (no hallucination beyond sources)
- **PDF export** :- Full report + Q&A history
- **Session history** :- Last 10 research sessions in sidebar

**Tech:** LangGraph · LangChain · Groq (Llama 3.3 70B) · DuckDuckGo Search · fpdf2

---

## Deployment (Streamlit Community Cloud)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select `app.py`
3. Under **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
4. Deploy — Streamlit Cloud installs `requirements.txt` automatically

---

## Constraints & Requirements

- **Team Size:** 3 Students
- **API Budget:** Free Tier Only (Groq free tier · DuckDuckGo no API key)
- **Framework:** LangGraph (mandatory)
- **Hosting:** Streamlit Community Cloud

---

## Technology Stack

| Component | Technology |
|:---|:---|
| **LLM** | Groq API — Llama 3.3 70B Versatile (free tier) |
| **Agentic Framework** | LangGraph |
| **Web Search** | DuckDuckGo Search (no API key) |
| **NLP (M1)** | spaCy · scikit-learn · Sumy · Gensim · NLTK |
| **UI** | Streamlit (multi-page) |
| **PDF Export** | fpdf2 |
| **Document Parsing** | PyMuPDF · python-docx |

---

### Evaluation

| Phase | Weight | Criteria |
|:---|:---|:---|
| **Mid-Sem (M1)** | 25% | NLP Pipeline correctness, Topic Modeling quality |
| **End-Sem (M2)** | 75% | Agentic workflow, report quality, UI, deployment |
