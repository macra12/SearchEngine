from flask import Flask, request, jsonify
from flask_cors import CORS
import math
import re
from collections import defaultdict
import numpy as np
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
import os
import logging
from logging.handlers import RotatingFileHandler
import json
from pathlib import Path

# Initialize app
app = Flask(__name__)
CORS(app)

# Constants
TFIDF_CACHE = 'tfidf_cache.pkl'
DEFAULT_SUGGESTIONS = [
    "machine learning",
    "deep learning",
    "natural language processing",
    "TF-IDF algorithm",
    "voice recognition",
    "information retrieval",
    "relevance ranking"
]

# Initialize NLP components
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
nltk.download('stopwords', quiet=True)

# Configure logging
handler = RotatingFileHandler('search_engine.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.ERROR)
app.logger.addHandler(handler)

# Document database (same as original)
def load_documents(json_path):
    """Load documents from a JSON file"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        # Validate the document structure
        required_keys = ['id', 'title', 'content', 'url', 'lastUpdated']
        for doc in documents:
            if not all(key in doc for key in required_keys):
                raise ValueError("Document missing required fields")
        
        return documents
    except Exception as e:
        app.logger.error(f"Failed to load documents: {str(e)}")
        raise

# Load documents from JSON file
try:
    # Use raw string for Windows paths and Path for cross-platform compatibility
    documents_path = Path(r"C:\Users\Admin\Documents\Life of Work\TGI\DSA\Project_Search_Engine\backend\data.json")
    documents = load_documents(documents_path)
except Exception as e:
    app.logger.critical(f"Failed to initialize documents: {str(e)}")
    # Optionally provide some default documents if the file fails to load
    documents = [
        {
            "id": 0,
            "title": "Error Loading Documents",
            "content": "The system failed to load the document database. Please check the data.json file.",
            "url": "/error",
            "lastUpdated": datetime.now().strftime("%Y-%m-%d")
        }
    ]


class SearchEngine:
    def __init__(self):
        self.documents = documents
        self.document_texts = [f"{doc['title']} {doc['content']}" for doc in documents]
        self.num_docs = len(self.document_texts)
        self.load_or_compute_tfidf()
        self.precompute_cooccurrences()
    
    def load_or_compute_tfidf(self):
        if os.path.exists(TFIDF_CACHE):
            try:
                with open(TFIDF_CACHE, 'rb') as f:
                    data = pickle.load(f)
                    self.vocabulary = data['vocabulary']
                    self.vocab_index = data['vocab_index']
                    self.tfidf_vectors = data['tfidf_vectors']
                    self.idf = data['idf']
                    self.word_doc_count = data['word_doc_count']
                    app.logger.info("Loaded TF-IDF from cache")
            except Exception as e:
                app.logger.error(f"Failed to load TF-IDF cache: {str(e)}")
                self.compute_tfidf()
        else:
            self.compute_tfidf()
    
    def compute_tfidf(self):
        app.logger.info("Computing TF-IDF vectors...")
        
        # Build vocabulary and document frequency
        self.vocabulary = set()
        self.word_doc_count = defaultdict(int)
        
        for text in self.document_texts:
            tokens = set(self.preprocess(text))
            for token in tokens:
                self.vocabulary.add(token)
                self.word_doc_count[token] += 1
        
        self.vocabulary = sorted(self.vocabulary)
        self.vocab_index = {word: idx for idx, word in enumerate(self.vocabulary)}
        
        # Compute IDF with smoothing
        self.idf = {}
        for word in self.vocabulary:
            self.idf[word] = math.log((self.num_docs + 1) / (1 + self.word_doc_count[word])) + 1
        
        # Compute TF-IDF vectors
        self.tfidf_vectors = np.zeros((self.num_docs, len(self.vocabulary)))
        
        for doc_idx, text in enumerate(self.document_texts):
            tokens = self.preprocess(text)
            word_count = defaultdict(int)
            for token in tokens:
                word_count[token] += 1
            
            for word, count in word_count.items():
                if word in self.vocab_index:
                    tf = count / len(tokens)
                    self.tfidf_vectors[doc_idx, self.vocab_index[word]] = tf * self.idf[word]
        
        # Normalize vectors
        norms = np.linalg.norm(self.tfidf_vectors, axis=1, keepdims=True)
        self.tfidf_vectors = np.divide(self.tfidf_vectors, norms, where=norms!=0)
        
        # Save to cache
        data = {
            'vocabulary': self.vocabulary,
            'vocab_index': self.vocab_index,
            'tfidf_vectors': self.tfidf_vectors,
            'idf': self.idf,
            'word_doc_count': self.word_doc_count
        }
        try:
            with open(TFIDF_CACHE, 'wb') as f:
                pickle.dump(data, f)
            app.logger.info("Saved TF-IDF to cache")
        except Exception as e:
            app.logger.error(f"Failed to save TF-IDF cache: {str(e)}")
    
    def precompute_cooccurrences(self):
        self.term_cooccur = defaultdict(lambda: defaultdict(int))
        for text in self.document_texts:
            tokens = set(self.preprocess(text))
            for token1 in tokens:
                for token2 in tokens:
                    if token1 != token2:
                        self.term_cooccur[token1][token2] += 1
    
    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        
        processed_tokens = []
        for token in tokens:
            if token not in stop_words:
                stemmed = stemmer.stem(token)
                processed_tokens.append(stemmed)
                
        return processed_tokens
    
    def search_tfidf(self, query, top_k=10):
        tokens = self.preprocess(query)
        if not tokens:
            return [], 0
        
        query_vec = np.zeros(len(self.vocabulary))
        
        # Compute query TF
        word_count = defaultdict(int)
        for token in tokens:
            word_count[token] += 1
        
        # Compute TF-IDF for query
        for word, count in word_count.items():
            if word in self.vocab_index:
                tf = count / len(tokens)
                query_vec[self.vocab_index[word]] = tf * self.idf.get(word, 0)
        
        # Normalize query vector
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec /= query_norm
        
        # Compute cosine similarity
        scores = np.dot(self.tfidf_vectors, query_vec)
        
        # Get all non-zero results
        results = []
        for idx in range(len(scores)):
            if scores[idx] > 0:
                results.append({
                    "doc_index": idx,
                    "score": float(scores[idx])
                })
        
        # Sort by score descending
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply top_k limit
        total_results = len(results)
        results = results[:top_k]
        
        # Format final results
        formatted_results = []
        for res in results:
            doc = self.documents[res['doc_index']]
            formatted_results.append({
                "id": doc["id"],
                "title": doc["title"],
                "url": doc["url"],
                "description": doc["content"][:150] + "...",
                "score": res["score"],
                "lastUpdated": doc["lastUpdated"]
            })
        
        return formatted_results, total_results
    
    def suggest(self, query):
        query = query.lower().strip()
        if not query:
            return DEFAULT_SUGGESTIONS[:7]
        
        query_terms = set(self.preprocess(query))
        suggestions = set()
        
        # Add co-occurring terms
        for term in query_terms:
            if term in self.term_cooccur:
                # Get top 3 co-occurring terms
                co_terms = sorted(self.term_cooccur[term].items(), 
                                key=lambda x: x[1], reverse=True)[:3]
                for co_term, _ in co_terms:
                    suggestions.add(f"{term} {co_term}")
        
        # Add matching titles
        for doc in self.documents:
            doc_text = f"{doc['title']} {doc['content']}".lower()
            if any(term in doc_text for term in query_terms):
                suggestions.add(doc['title'])
        
        # Fallback to default if no good suggestions
        if not suggestions:
            return DEFAULT_SUGGESTIONS[:7]
        
        return list(suggestions)[:7]

# Initialize search engine
search_engine = SearchEngine()

# API endpoints
@app.route('/api/health')
def health_check():
    return jsonify({
        "status": "ok", 
        "message": "Search engine backend is running",
        "timestamp": datetime.now().isoformat(),
        "documents": len(search_engine.documents)
    })

@app.route('/api/search', methods=['GET'])
def search():
    query = request.args.get('q', '').strip()
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 10, type=int)
    
    if not query:
        return jsonify({
            "results": [],
            "total": 0,
            "page": page,
            "total_pages": 0
        })
    
    try:
        # Get all results
        all_results, total_results = search_engine.search_tfidf(query, top_k=100)
        
        # Apply pagination
        total_pages = max(1, math.ceil(total_results / limit))
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_results = all_results[start_idx:end_idx]
        
        return jsonify({
            "results": paginated_results,
            "total": total_results,
            "page": page,
            "total_pages": total_pages
        })
    except Exception as e:
        app.logger.error(f"Search error for query '{query}': {str(e)}")
        return jsonify({
            "error": "Search failed", 
            "message": "Internal server error"
        }), 500

@app.route('/api/suggest', methods=['GET'])
def suggest():
    try:
        query = request.args.get('q', '').strip()
        suggestions = search_engine.suggest(query)
        return jsonify({"suggestions": suggestions})
    except Exception as e:
        app.logger.error(f"Suggestion error for query '{query}': {str(e)}")
        return jsonify({"suggestions": DEFAULT_SUGGESTIONS[:7]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)