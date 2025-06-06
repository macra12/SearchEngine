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

# Download NLTK resources
nltk.download('stopwords', quiet=True)

app = Flask(__name__)
CORS(app)

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Enhanced document database
documents = [
    {
        "id": 1, 
        "title": "Introduction to Machine Learning", 
        "content": "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. It includes supervised, unsupervised, and reinforcement learning techniques.",
        "url": "/doc1",
        "lastUpdated": "2024-03-15"
    },
    {
        "id": 2, 
        "title": "Deep Learning Fundamentals", 
        "content": "Deep learning uses neural networks with multiple layers to learn complex patterns in large amounts of data. Applications include image recognition and natural language processing.",
        "url": "/doc2",
        "lastUpdated": "2024-04-22"
    },
    {
        "id": 3, 
        "title": "Natural Language Processing Techniques", 
        "content": "NLP enables computers to understand, interpret, and generate human language through machine learning. Key techniques include tokenization, stemming, and named entity recognition.",
        "url": "/doc3",
        "lastUpdated": "2024-02-10"
    },
    {
        "id": 4, 
        "title": "TF-IDF Algorithm Explained", 
        "content": "TF-IDF is a statistical measure that evaluates how relevant a word is to a document in a collection. It combines term frequency with inverse document frequency.",
        "url": "/doc4",
        "lastUpdated": "2024-05-05"
    },
    {
        "id": 5, 
        "title": "Voice Recognition Systems", 
        "content": "Modern voice recognition systems use deep learning to convert spoken language into text with high accuracy. They employ neural networks trained on large speech datasets.",
        "url": "/doc5",
        "lastUpdated": "2024-01-18"
    },
    {
        "id": 6, 
        "title": "Information Retrieval Systems", 
        "content": "Information retrieval systems help users find relevant information from large collections. They use ranking algorithms like BM25 and vector space models.",
        "url": "/doc6",
        "lastUpdated": "2024-03-28"
    },
    {
        "id": 7, 
        "title": "Search Engine Architecture", 
        "content": "Modern search engines consist of crawlers, indexers, and query processors. They handle billions of queries daily using distributed systems.",
        "url": "/doc7",
        "lastUpdated": "2024-04-05"
    },
    {
        "id": 8, 
        "title": "Relevance Ranking Algorithms", 
        "content": "Relevance ranking determines the order of search results. Popular algorithms include PageRank, BM25, and neural ranking models.",
        "url": "/doc8",
        "lastUpdated": "2024-02-20"
    },
    {
        "id": 9, 
        "title": "Text Preprocessing Techniques", 
        "content": "Text preprocessing includes tokenization, stopword removal, stemming, and lemmatization. These techniques prepare text for machine learning.",
        "url": "/doc9",
        "lastUpdated": "2024-05-12"
    },
    {
        "id": 10, 
        "title": "Vector Space Models in NLP", 
        "content": "Vector space models represent documents and queries as vectors in high-dimensional space. Similarity is measured using cosine similarity.",
        "url": "/doc10",
        "lastUpdated": "2024-01-30"
    }
]

# Precompute TF-IDF vectors
document_texts = [f"{doc['title']} {doc['content']}" for doc in documents]
vocabulary = set()
word_doc_count = defaultdict(int)

# Enhanced preprocessing with stemming and stopword removal
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    
    # Remove stopwords and apply stemming
    processed_tokens = []
    for token in tokens:
        if token not in stop_words:
            stemmed = stemmer.stem(token)
            processed_tokens.append(stemmed)
            
    return processed_tokens

# Build vocabulary and document frequency
for text in document_texts:
    tokens = set(preprocess(text))
    for token in tokens:
        vocabulary.add(token)
        word_doc_count[token] += 1

vocabulary = sorted(vocabulary)
vocab_index = {word: idx for idx, word in enumerate(vocabulary)}
num_docs = len(document_texts)

# Compute IDF with smoothing
idf = {}
for word in vocabulary:
    idf[word] = math.log((num_docs + 1) / (1 + word_doc_count[word])) + 1

# Compute TF-IDF vectors
tfidf_vectors = np.zeros((num_docs, len(vocabulary)))

for doc_idx, text in enumerate(document_texts):
    tokens = preprocess(text)
    word_count = defaultdict(int)
    for token in tokens:
        word_count[token] += 1
    
    for word, count in word_count.items():
        if word in vocab_index:
            tf = count / len(tokens)
            tfidf_vectors[doc_idx, vocab_index[word]] = tf * idf[word]

# Normalize vectors
norms = np.linalg.norm(tfidf_vectors, axis=1, keepdims=True)
tfidf_vectors = np.divide(tfidf_vectors, norms, where=norms!=0)

def search_tfidf(query, top_k=10):
    tokens = preprocess(query)
    if not tokens:
        return [], 0
    
    query_vec = np.zeros(len(vocabulary))
    
    # Compute query TF
    word_count = defaultdict(int)
    for token in tokens:
        word_count[token] += 1
    
    # Compute TF-IDF for query
    for word, count in word_count.items():
        if word in vocab_index:
            tf = count / len(tokens)
            query_vec[vocab_index[word]] = tf * idf.get(word, 0)
    
    # Normalize query vector
    query_norm = np.linalg.norm(query_vec)
    if query_norm > 0:
        query_vec /= query_norm
    
    # Compute cosine similarity
    scores = np.dot(tfidf_vectors, query_vec)
    
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
        doc = documents[res['doc_index']]
        formatted_results.append({
            "title": doc["title"],
            "url": doc["url"],
            "description": doc["content"][:150] + "...",
            "score": res["score"],
            "lastUpdated": doc["lastUpdated"]
        })
    
    return formatted_results, total_results

@app.route('/api/health')
def health_check():
    return jsonify({
        "status": "ok", 
        "message": "Search engine backend is running",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/search', methods=['GET'])
def search():
    query = request.args.get('q', '')
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
        all_results, total_results = search_tfidf(query, top_k=100)
        
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
        app.logger.error(f"Search error: {str(e)}")
        return jsonify({
            "error": "Search failed", 
            "message": "Internal server error"
        }), 500

@app.route('/api/suggest', methods=['GET'])
def suggest():
    try:
        query = request.args.get('q', '')
        suggestions = []
        
        if query:
            # Find matching terms in vocabulary
            query_terms = set(preprocess(query))
            for term in query_terms:
                if term in vocabulary:
                    # Add similar terms from vocabulary
                    suggestions.extend([
                        f"{term} learning",
                        f"{term} algorithm",
                        f"{term} techniques",
                        f"advanced {term}",
                        f"{term} systems"
                    ])
            
            # Add document titles that contain query terms
            for doc in documents:
                doc_text = f"{doc['title']} {doc['content']}".lower()
                if any(term in doc_text for term in query_terms):
                    suggestions.append(doc['title'])
            
            # Remove duplicates and limit to 7
            suggestions = list(set(suggestions))[:7]
        else:
            # Return popular searches
            suggestions = [
                "machine learning",
                "deep learning",
                "natural language processing",
                "TF-IDF algorithm",
                "voice recognition",
                "information retrieval",
                "relevance ranking"
            ]
        
        return jsonify({"suggestions": suggestions})
    except Exception as e:
        app.logger.error(f"Suggestion error: {str(e)}")
        return jsonify({"suggestions": []})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)