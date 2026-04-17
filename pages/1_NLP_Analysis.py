import streamlit as st
import pandas as pd
from src.data_fetcher import extract_text_from_pdf, extract_text_from_txt, extract_text_from_docx, fetch_wikipedia_content
from src.nlp_pipeline import clean_text, extract_keywords_tfidf, perform_topic_modeling, generate_extractive_summary

st.set_page_config(page_title="NLP Research Analysis System", layout="wide")

st.title("NLP Research Analysis System (Milestone 1)")
st.markdown("Analyze documents or search topics to extract themes, keywords, and structured summaries using traditional NLP.")

st.sidebar.header("Input Section")
input_method = st.sidebar.radio("Choose Input Method", ["Upload Document(s)", "Search Topic (Wikipedia)"])

raw_text = ""
source_info = ""

if input_method == "Upload Document(s)":
    uploaded_files = st.sidebar.file_uploader("Upload PDF, DOCX or TXT files", type=["pdf", "txt", "docx"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            file_type = file.name.split('.')[-1].lower()
            if file_type == 'pdf':
                raw_text += extract_text_from_pdf(file) + "\n\n"
            elif file_type == 'docx':
                raw_text += extract_text_from_docx(file) + "\n\n"
            elif file_type == 'txt':
                raw_text += extract_text_from_txt(file) + "\n\n"
        source_info = f"Uploaded {len(uploaded_files)} document(s)"

elif input_method == "Search Topic (Wikipedia)":
    search_query = st.sidebar.text_input("Enter a Research Topic (e.g., Quantum Computing)")
    if st.sidebar.button("Fetch & Analyze"):
        with st.spinner("Fetching content from Wikipedia..."):
            if search_query:
                raw_text = fetch_wikipedia_content(search_query)
                if not raw_text:
                    st.sidebar.error("Could not fetch topic. Try a different keyword.")
                else:
                    source_info = f"Wikipedia Search: {search_query}"
            else:
                st.sidebar.warning("Please enter a topic.")

if raw_text:
    st.success(f"Successfully loaded content. Source: {source_info}")

    with st.spinner("Cleaning text data..."):
        cleaned_text = clean_text(raw_text)

    tab1, tab2, tab3, tab4 = st.tabs(["Extractive Summary", "Keywords (TF-IDF)", "Topic Modeling (LDA)", "Raw Text Preview"])

    with tab1:
        st.subheader("Extractive Summary")
        st.markdown("*Summary generated using TextRank algorithm scoring sentences.*")
        with st.spinner("Generating summary..."):
            summary = generate_extractive_summary(cleaned_text, sentences_count=7)
            if summary:
                st.info(summary)
            else:
                st.warning("Not enough text to generate a meaningful summary.")

    with tab2:
        st.subheader("Key Themes and Keywords (TF-IDF)")
        with st.spinner("Extracting keywords..."):
            keywords = extract_keywords_tfidf(cleaned_text, top_n=15)
            if keywords:
                df_keywords = pd.DataFrame(keywords, columns=["Keyword", "TF-IDF Score"])
                st.dataframe(df_keywords.style.background_gradient(cmap="Blues"), use_container_width=True)

                st.bar_chart(df_keywords.set_index("Keyword"))
            else:
                st.warning("Failed to extract keywords. Text might be too short.")

    with tab3:
        st.subheader("Topic Clusters (LDA)")
        st.markdown("*Identified underlying topics using Latent Dirichlet Allocation (LDA).*")
        with st.spinner("Running Topic Modeling..."):
            topics = perform_topic_modeling(cleaned_text, n_topics=4, n_words=6)
            if topics:
                for idx, topic_words in enumerate(topics):
                    st.write(f"**Topic {idx + 1}:** {', '.join(topic_words)}")
            else:
                st.warning("Insufficient text volume for effective topic modeling.")

    with tab4:
        st.subheader("Raw Text Snippet")
        st.text_area("Snippet of original content:", value=raw_text[:2000] + "...\n\n[TRUNCATED]", height=300, disabled=True)

else:
    st.info("Please select an input method and provide data from the sidebar to begin analysis.")
