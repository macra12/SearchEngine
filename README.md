# TF-IDF Search Engine

## Introduction
This project implements a search engine using **Term Frequency-Inverse Document Frequency (TF-IDF)**, a fundamental algorithm in information retrieval. The engine ranks documents based on their relevance to user queries by analyzing the importance of terms in a document relative to a corpus.

## Problem Statement
Traditional keyword searches often return irrelevant results due to:
- Ignoring term specificity across documents
- Overemphasizing frequent but insignificant words
- Lacking nuanced document ranking  
This project solves these issues through TF-IDF's statistical approach to quantify term relevance.

## Project Vision
To create an efficient, scalable search engine that:
1. Delivers contextually relevant results
2. Handles diverse text datasets
3. Demonstrates core IR principles transparently

## Solution
The TF-IDF algorithm calculates relevance using:
- **Term Frequency (TF)**: Frequency of a term in a document
- **Inverse Document Frequency (IDF)**: Rarity of a term across all documents  
Relevance Score = TF * IDF  
Higher scores indicate greater document relevance to the query.

## Technology Stack
- **Language**: Python 3
- **Key Libraries**: 
  - `scikit-learn` (TfidfVectorizer)
  - `numpy` (numerical operations)
  - `pandas` (data handling)
- **Data**: Text documents (e.g., .txt, .csv files)
- **Interface**: Command-line or basic web frontend (optional)

## Development Process
1. **Data Preprocessing**  
   - Text cleaning (lowercasing, removing punctuation)
   - Tokenization and stopword removal
2. **TF-IDF Implementation**  
   - Document-term matrix generation
   - TF and IDF calculations
3. **Query Processing**  
   - Vectorizing user queries
   - Computing cosine similarity between query and documents
4. **Ranking & Output**  
   - Sorting documents by relevance score
   - Displaying top-k results

## Project Scope
### Included
- Core TF-IDF algorithm implementation
- Document indexing and ranking
- Query processing module
- Scalable design for medium-sized datasets

### Excluded
- Real-time indexing of dynamic data
- Advanced NLP techniques (stemming/lemmatization)
- User authentication or persistent storage

## Author
**Khin Nara**  
