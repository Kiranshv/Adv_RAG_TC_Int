# Advanced RAG Explorer

End-to-end teaching demo for The Testing Academy over a large VWO test case corpus.

## Highlights

- Hybrid retrieval with `BAAI/bge-m3` (dense + sparse from one model)
- Qdrant vector DB using embedded mode by default (`./qdrant_data/`)
- Re-ranking with `BAAI/bge-reranker-v2-m3`
- Query rewriting with OpenRouter (or Groq as alternative)
- Grounded generation with chunk citations
- Claude-inspired warm UI with two-pane pipeline + content/chat view

## Pipeline

Stage 1 (Ingest):
CSV/XLSX -> rows -> docs -> chunk -> bge-m3 dense+sparse -> Qdrant `vwo_test_cases`

Stage 2 (Chat):
Question -> rewrite -> hybrid search -> RRF fuse -> rerank -> grounded answer

## Setup

```bash
cd Advanced_RAG/Advance_RAG_EXPLAIN
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Set `.env` keys for either OpenRouter or Groq.

## Run

```bash
.venv\Scripts\activate
python app.py
```

Open: http://127.0.0.1:5050

## Optional CLI ingestion

```bash
python ingest.py testcase/vwo_test_cases_jira_5000.csv \
  --text-cols title,steps,expected,tags \
  --meta-cols id,jira_id,priority,module
```

## Pages

- `/upload`: upload + preview + column selection
- `/ingest`: SSE live stage tracker
- `/chunks`: chunk viewer with filters
- `/chat`: rewrite + hybrid retrieval + rerank + grounded output
- `/static/rag_explorer.html`: animated architecture explainer

## Notes

- First model load is slow due Hugging Face downloads.
- Embedded Qdrant means no Docker needed.
- For external Qdrant, set `QDRANT_URL=http://host:6333`.
