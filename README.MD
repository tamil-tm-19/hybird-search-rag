Hybrid Search RAG Pipeline

A retrieval-augmented generation (RAG) system that combines dense (semantic) and sparse (keyword) search using Reciprocal Rank Fusion, followed by cross-encoder reranking, to answer questions grounded in real-world documents.

Tested on the Tesla 2025 10-K annual report.

Why Hybrid Search?

Dense (embedding) search is great at capturing semantic meaning but can miss exact keyword matches (numbers, names, IDs). Sparse (BM25) search is great at exact term matching but misses semantic similarity. Combining both — instead of relying on either alone — improves retrieval recall and precision, especially on dense technical documents like financial reports.

Architecture
                     ┌─────────────┐
                     │  User Query │
                     └──────┬──────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
     ┌─────────────────┐        ┌──────────────────┐
     │ Dense Embedding  │        │   BM25 Encoder    │
     │ (SentenceTransf.)│        │  (sparse vector)  │
     └────────┬─────────┘        └─────────┬─────────┘
              ▼                           ▼
     ┌─────────────────┐        ┌──────────────────┐
     │  Pinecone Query  │        │  Pinecone Query   │
     │  (dense vector)  │        │  (sparse vector)  │
     └────────┬─────────┘        └─────────┬─────────┘
              └─────────────┬─────────────┘
                            ▼
                  ┌───────────────────┐
                  │   RRF Fusion       │
                  │ (rank-based merge) │
                  └─────────┬──────────┘
                            ▼
                  ┌───────────────────┐
                  │ CrossEncoder       │
                  │ Reranking (Cohere) │
                  └─────────┬──────────┘
                            ▼
                  ┌───────────────────┐
                  │   Groq LLM         │
                  │   Generation       │
                  └─────────┬──────────┘
                            ▼
                       Final Answer
Pipeline Stages
Document Parsing — PDF parsed page-wise using PyMuPDF (langchain_community.PyMuPDFLoader)
Chunking — Recursive character text splitting with configurable chunk size/overlap, preserving page-level metadata
Embedding — Dense vectors via SentenceTransformer (all-MiniLM-L6-v2, 384-dim)
Sparse Indexing — BM25 encoder (pinecone-text) fit once on the full corpus, stored/loaded for reuse
Vector Store — Pinecone serverless index, storing both dense values and sparse values per chunk
Retrieval — Two parallel Pinecone queries (dense-only, sparse-only) per user query
Fusion — Reciprocal Rank Fusion (RRF) merges both ranked lists by position, not raw score — avoids score-scale mismatch between cosine similarity and BM25 scores
Reranking — Cohere CrossEncoder reranks the fused candidate set for final relevance ordering
Generation — Groq LLM (openai/gpt-oss-120b) generates a grounded answer strictly from retrieved context, with a system prompt enforcing no hallucination beyond provided context
Tech Stack
Component	Tool
Document parsing	PyMuPDF (via LangChain loaders)
Chunking	LangChain RecursiveCharacterTextSplitter
Dense embeddings	SentenceTransformers (all-MiniLM-L6-v2)
Sparse encoding	BM25 (pinecone-text)
Vector database	Pinecone (serverless, dense + sparse)
Fusion	Reciprocal Rank Fusion (custom implementation)
Reranking	Cohere Rerank API
Generation	Groq API (LLM inference)
Language	Python 3.10+
Project Structure
hybrid-search-rag/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   └── sample.pdf
├── src/
│   ├── document_loader.py    # PDF parsing
│   ├── chunking.py           # Recursive chunking + chunk ID assignment
│   ├── embeddings.py         # Dense embedding model wrapper
│   ├── bm25_index.py         # Sparse BM25 encoder (fit/load/encode)
│   ├── vector_store.py       # Pinecone upsert + dual query
│   ├── fusion.py             # RRF fusion logic
│   ├── reranker.py           # Cohere CrossEncoder reranking
│   └── generation.py         # Groq LLM answer generation
└── app.py                    # Entry point
How to Run
bash
git clone https://github.com/tamil-tm-19/hybird-search-rag.git
cd hybird-search-rag
pip install -r requirements.txt
cp .env.example .env   # add your PINECONE_API_KEY, GROQ_API_KEY, COHERE_API_KEY
python app.py
Sample Query

Query: "What was Tesla's total automotive revenue in 2024?"

Answer: (grounded strictly in retrieved 10-K chunks — no outside knowledge, numbers reproduced exactly as reported)

Key Design Decisions & What I Learned
RRF over weighted score fusion — dense (cosine) and sparse (BM25) scores live on incompatible scales; rank-based fusion avoids manual weight tuning
BM25 must be fit once on the full corpus, not per batch — IDF is a global corpus statistic; fitting on partial batches during upsert corrupts term rarity scores for previously indexed documents. Fixed this after finding it during a code review pass.
Retrieve-then-rerank pattern — reranking the entire corpus per query is too expensive (cross-encoders score query+document pairs jointly with no precomputed vectors); fusing first narrows to a small candidate set before the expensive reranking step
Fixed a fusion bug where the RRF merge loop returned after processing only the first ranked list, silently degrading "hybrid" search into dense-only search
Future Improvements
Formal RAGAS evaluation (faithfulness, context precision/recall)
Semantic/structure-aware chunking (filter TOC and boilerplate sections)
FastAPI + Docker deployment
Logging (replace print statements) and unit tests for fusion/BM25 logic
Metadata filtering combined with hybrid search