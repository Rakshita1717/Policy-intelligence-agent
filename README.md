# Mini Autonomous Policy Intelligence Agent

## Overview
This project builds a lightweight AI-driven Policy Intelligence Agent using the Sentiment140 dataset as an offline proxy for X (Twitter).

The system performs:
- Data processing and keyword filtering
- Sentiment analysis
- Topic modeling
- Embedding generation
- FAISS vector storage
- RAG-based retrieval
- Agent routing
- Executive report generation

## Architecture
Data → Preprocessing → Sentiment → Topics → Embeddings → FAISS → RAG → Agent → Report

## Models Used
- TF-IDF + Logistic Regression (Sentiment)
- LDA (Topic Modeling)
- SentenceTransformers (Embeddings)
- FAISS (Vector Search)

## Agent Logic
- If query contains "sentiment" → Sentiment summary
- If query contains "topic" → Topic exploration
- Else → RAG retrieval

## How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Run:
   python main.py

## Notes
- Sentiment140 is used as an offline proxy.
- No live APIs are used.