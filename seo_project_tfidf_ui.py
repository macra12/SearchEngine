import streamlit as st
import math
import re
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict
import pandas as pd
import plotly.express as px
import time
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import aiohttp
import asyncio
import io

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Streamlit page configuration
st.set_page_config(page_title="SEO TF-IDF Analyzer", layout="wide", page_icon="🔍")

@st.cache_data
def fetch_url_content(url):
    """Fetch content from a URL (cached for performance)."""
    try:
        if "youtube.com/watch" in url:
            return scrape_youtube(url)
        else:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            content = soup.find_all(['p', 'h1', 'h2', 'h3', 'meta'])
            text = ' '.join([element.get_text() for element in content if element.get_text()])
            return text.strip() if text.strip() else None
    except Exception as e:
        return str(e)

async def async_scrape_url(url, session):
    """Asynchronously scrape a URL."""
    try:
        if "youtube.com/watch" in url:
            return url, scrape_youtube(url)
        else:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            async with session.get(url, headers=headers, timeout=10) as response:
                response.raise_for_status()
                text = await response.text()
                soup = BeautifulSoup(text, 'html.parser')
                content = soup.find_all(['p', 'h1', 'h2', 'h3', 'meta'])
                text = ' '.join([element.get_text() for element in content if element.get_text()])
                return url, text.strip() if text.strip() else None
    except Exception as e:
        return url, str(e)

def scrape_youtube(url):
    """Scrape YouTube transcript or metadata."""
    try:
        video_id = url.split("v=")[1].split("&")[0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry['text'] for entry in transcript])
        return text.strip() if text.strip() else None
    except (TranscriptsDisabled, NoTranscriptFound):
        # Fallback to metadata
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.find('title').get_text() if soup.find('title') else ''
            description = soup.find('meta', {'name': 'description'})['content'] if soup.find('meta', {'name': 'description'}) else ''
            keywords = soup.find('meta', {'name': 'keywords'})['content'] if soup.find('meta', {'name': 'keywords'}) else ''
            text = f"{title} {description} {keywords}"
            return text.strip() if text.strip() else None
        except Exception as e:
            return str(e)

class SEODataCollector:
    def __init__(self):
        self.documents = []
        self.urls = []
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def add_document(self, document, url):
        if document:
            self.documents.append(document)
            self.urls.append(url)

    def preprocess_text(self, text):
        if not isinstance(text, str):
            return []
        tokens = word_tokenize(text.lower())
        tokens = [self.stemmer.stem(token) for token in tokens 
                  if token.isalnum() and token not in self.stop_words]
        return tokens

class TFIDF_SEO:
    def __init__(self):
        self.documents = []
        self.word_doc_freq = defaultdict(int)
        self.tf_idf_scores = []
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def preprocess_text(self, text):
        if not isinstance(text, str):
            return []
        tokens = word_tokenize(text.lower())
        tokens = [self.stemmer.stem(token) for token in tokens 
                  if token.isalnum() and token not in self.stop_words]
        return tokens

    def add_document(self, document):
        if document:
            self.documents.append(document)
            tokens = set(self.preprocess_text(document))
            for token in tokens:
                self.word_doc_freq[token] += 1

    def calculate_tf(self, document):
        tokens = self.preprocess_text(document)
        word_count = len(tokens)
        tf_scores = defaultdict(float)
        for token in tokens:
            tf_scores[token] += 1.0 / (word_count + 1)
        return tf_scores

    def calculate_idf(self):
        num_docs = len(self.documents)
        idf_scores = {}
        for word, doc_count in self.word_doc_freq.items():
            idf_scores[word] = math.log((num_docs + 1) / (doc_count + 1)) + 1
        return idf_scores

    def calculate_tfidf(self):
        self.tf_idf_scores = []
        idf_scores = self.calculate_idf()
        for doc in self.documents:
            tf_scores = self.calculate_tf(doc)
            doc_tfidf = {}
            for word, tf in tf_scores.items():
                doc_tfidf[word] = tf * idf_scores.get(word, 0)
            self.tf_idf_scores.append(doc_tfidf)

    def search(self, query):
        query_tokens = self.preprocess_text(query)
        query_tf = defaultdict(float)
        for token in query_tokens:
            query_tf[token] += 1.0 / (len(query_tokens) + 1)

        scores = []
        for doc_idx, doc_tfidf in enumerate(self.tf_idf_scores):
            score = 0
            for token in query_tokens:
                score += query_tf[token] * doc_tfidf.get(token, 0)
            scores.append((score, doc_idx))
        scores.sort(reverse=True)
        return scores

    def get_top_keywords(self, doc_idx, top_n=5):
        if doc_idx < len(self.tf_idf_scores):
            doc_tfidf = self.tf_idf_scores[doc_idx]
            sorted_keywords = sorted(doc_tfidf.items(), key=lambda x: x[1], reverse=True)
            return sorted_keywords[:top_n]
        return []

    def calculate_keyword_density(self, document):
        tokens = self.preprocess_text(document)
        total_words = len(tokens)
        word_counts = defaultdict(int)
        for token in tokens:
            word_counts[token] += 1
        density = {word: (count / total_words) * 100 for word, count in word_counts.items()}
        sorted_density = sorted(density.items(), key=lambda x: x[1], reverse=True)[:5]
        return sorted_density

# Streamlit UI
def main():
    st.title("🔍 Ultimate SEO TF-IDF Analyzer")
    st.markdown("""
        The best SEO tool to analyze websites, YouTube videos, or text content using TF-IDF. 
        Add URLs, upload files, or use sample data, then optimize your content with actionable insights.
    """)

    # Initialize session state
    if 'collector' not in st.session_state:
        st.session_state.collector = SEODataCollector()
        st.session_state.seo_analyzer = TFIDF_SEO()
        st.session_state.urls = []
        st.session_state.documents = []

    # Input Section
    st.header("Input Data")
    input_method = st.radio("Choose input method:", ("Enter URLs", "Upload Text/File", "Use Sample Data"))

    if input_method == "Enter URLs":
        urls_input = st.text_area("Enter URLs (one per line, supports YouTube):", 
                                  "https://moz.com/learn/seo\nhttps://www.youtube.com/watch?v=HNuiDXuNU7E")
        if st.button("Scrape URLs"):
            with st.spinner("Scraping content..."):
                urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
                async def scrape_all():
                    async with aiohttp.ClientSession() as session:
                        tasks = [async_scrape_url(url, session) for url in urls]
                        return await asyncio.gather(*tasks)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(scrape_all())
                for url, content in results:
                    if content and not isinstance(content, str) or (isinstance(content, str) and not content.startswith("Error")):
                        st.session_state.collector.add_document(content, url)
                        st.session_state.seo_analyzer.add_document(content)
                        st.success(f"Content scraped from {url}")
                    else:
                        st.warning(f"Failed to scrape {url}: {content}")
            st.success("Scraping completed!")

    elif input_method == "Upload Text/File":
        st.subheader("Upload Text or File")
        uploaded_text = st.text_area("Enter or paste text content:")
        uploaded_file = st.file_uploader("Upload a text or CSV file", type=["txt", "csv"])
        if st.button("Add Content"):
            if uploaded_text:
                st.session_state.collector.add_document(uploaded_text, "Uploaded Text")
                st.session_state.seo_analyzer.add_document(uploaded_text)
                st.success("Text content added!")
            if uploaded_file:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    text = ' '.join(df.astype(str).values.flatten())
                else:
                    text = uploaded_file.read().decode('utf-8')
                st.session_state.collector.add_document(text, uploaded_file.name)
                st.session_state.seo_analyzer.add_document(text)
                st.success(f"File {uploaded_file.name} added!")

    else:  # Use Sample Data
        if st.button("Load Sample Data"):
            sample_docs = [
                "SEO optimization improves website ranking on search engines",
                "Search engine optimization involves keywords and content strategy"
            ]
            for i, doc in enumerate(sample_docs):
                st.session_state.collector.add_document(doc, f"Sample Document {i+1}")
                st.session_state.seo_analyzer.add_document(doc)
            st.success("Sample data loaded!")

    # Search Query
    st.header("Search Query")
    st.info("Enter keywords or text to match with the documents (e.g., YouTube metadata or transcript).")
    search_query = st.text_input("Enter search query:", "SEO optimization techniques")

    # Analyze Button
    if st.button("Analyze"):
        if st.session_state.documents:
            with st.spinner("Calculating TF-IDF scores..."):
                st.session_state.seo_analyzer.calculate_tfidf()
                results = st.session_state.seo_analyzer.search(search_query)

            # Display Results
            st.header("Analysis Results")
            st.subheader(f"Search Query: {search_query}")
            st.write("**Ranked Documents:**")
            data = []
            for score, doc_idx in results:
                top_keywords = st.session_state.seo_analyzer.get_top_keywords(doc_idx)
                keyword_density = st.session_state.seo_analyzer.calculate_keyword_density(st.session_state.documents[doc_idx])
                keywords = ", ".join([f"{k} ({v:.4f})" for k, v in top_keywords])
                densities = ", ".join([f"{k} ({v:.2f}%)" for k, v in keyword_density])
                data.append({
                    "Document": f"Doc {doc_idx + 1} ({st.session_state.urls[doc_idx]})",
                    "Score": f"{score:.4f}",
                    "Top Keywords (TF-IDF)": keywords,
                    "Keyword Density": density,
                    "Content Preview": st.session_state.documents[doc_idx][:100] + "..."
                })
            df_results = pd.DataFrame(data)
            st.dataframe(df_results, use_container_width=True)

            # Download Button
            csv = df_results.to_csv(index=False)
            st.download_button("Download Results as CSV", csv, "seo_results.csv", "text/csv")

            # Visualizations
            if results:
                top_doc_idx = results[0][1]
                with st.expander("Visualizations", expanded=True):
                    # TF-IDF Chart
                    top_keywords = st.session_state.seo_analyzer.get_top_keywords(top_doc_idx)
                    if top_keywords:
                        keywords, scores = zip(*top_keywords)
                        df_tfidf = pd.DataFrame({"Keyword": keywords, "TF-IDF Score": scores})
                        fig_tfidf = px.bar(df_tfidf, x="Keyword", y="TF-IDF Score", 
                                         title="Top Keywords by TF-IDF Score",
                                         color="TF-IDF Score", color_continuous_scale="Blues")
                        st.plotly_chart(fig_tfidf, use_container_width=True)

                    # Keyword Density Chart
                    keyword_density = st.session_state.seo_analyzer.calculate_keyword_density(st.session_state.documents[top_doc_idx])
                    if keyword_density:
                        keywords, densities = zip(*keyword_density)
                        df_density = pd.DataFrame({"Keyword": keywords, "Density (%)": density})
                        fig_density = px.pie(df_density, names="Keyword", values="Density (%)", 
                                           title="Keyword Density Distribution")
                        st.plotly_chart(fig_density, use_container_width=True)

            # SEO Recommendations
            st.subheader("SEO Recommendations")
            if results:
                top_keywords = st.session_state.seo_analyzer.get_top_keywords(results[0][1])
                if top_keywords:
                    keywords = [k for k, _ in top_keywords]
                    st.markdown(f"""
                        - **Optimize Meta Tags**: Include top keywords ({', '.join(keywords[:3])}) in title, description, and H1 tags.
                        - **Content Strategy**: Create content around high TF-IDF terms to improve relevance.
                        - **Internal Linking**: Use keywords as anchor text for internal links.
                        - **Monitor Performance**: Track rankings with tools like Google Search Console or Ahrefs.
                    """)

        else:
            st.error("No documents available. Please add URLs, text, files, or use sample data.")

# Custom CSS for modern look
st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
    }
    .stTextInput>div>input {
        border: 2px solid #1f77b4;
        border-radius: 8px;
    }
    .stTextArea>div>textarea {
        border: 2px solid #1f77b4;
        border-radius: 8px;
    }
    h1, h2, h3 {
        color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()