# Math ArXiv RAG Bot

FastAPI-based Retrieval-Augmented Generation (RAG) bot that answers questions about recent **Number Theory (math.NT)** and **Algebraic Geometry (math.AG)** papers on arXiv.  
Pipeline: **ingest → embed (BGE-M3 ONNX) → Qdrant vector search → claim extraction (LLM) → cited answer**.

## Features
- **CPU-only** embeddings via **BGE-M3 ONNX** (no PyTorch required).
- **Qdrant** vector DB (run via Docker in one command).
- ar5iv HTML parsing → section-aware, overlapping text chunking.
- LLM prompts enforce grounded, bracket-cited answers.

---

## Requirements
- macOS/Linux (tested on macOS).
- Python **3.9** (virtualenv recommended).
- **Docker** (for Qdrant).
- OpenAI account/key (used **only** for claims + final answer; embeddings are local ONNX by default).

---

## Setup

```bash
# 1) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3) Create your environment file
cp .env.example .env
# open .env and fill placeholders
```

## Test run / smoke test

Two ways to test the bot:

1) **Retrieval-only** (no LLM key required) – verifies Qdrant + ONNX embeddings + ingestion.
2) **End-to-end** (LLM required) – verifies the `/ask` API returns a cited answer.

> Tip: You can watch Qdrant’s data grow in **http://localhost:6333/dashboard**.

### A. Index a tiny batch (~5 papers)

Open one terminal with Qdrant running (see above), then in another terminal:

```bash
# from repo root, with .venv active
python -c 'import ingest_math; ingest_math.run(max_results=5)'
```

You should see logs like:
```
Indexed 2508.21xxxv1 with 123 chunks
Indexed 2508.20xxxv1 with 98 chunks
...
```

### B. Retrieval-only sanity check:
This is to verify that;
1. The ONNX embedder runs locally,
2. The query is embedded,
3. Qdrant returns passage payloads.
```
python - <<'PY'
import asyncio
from retrieval import retrieve_passages

async def main():
    res = await retrieve_passages("Néron–Severi group and Picard group relation", limit=5)
    print("Top results (score, arxiv_id, section):")
    for p in res:
        print(round(p["score"], 3), p["arxiv_id"], "-", p["section"])
        print("  ", (p["text"][:140] + "...").replace("\n"," "))
        print("  ", p["source_html"])
        print()

asyncio.run(main())
PY
```
If you get passages but they look like ar5iv UI (e.g., “View PDF”, “Submission history”), that’s expected for now—see Troubleshooting on filtering junk in ingest_math.py

### C. End-to-end test (LLM Required)
Calls the FastAPI endpoint that performs retrieval → claim extraction (LLM) → grounded answer composition (LLM).
a. Start the API
  uvicorn server:app --reload
b. In another terminal, POST a question:
```
  curl -sS -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Explain the relation between the Picard group and the Néron–Severi group."}' \
  | python -m json.tool
```
c. You should see a JSON response of the form:
```
  {
  "answer": "Grounded statements related to ... [arXiv:..., Section] ...",
  "claims": [
    {"claim":"...", "arxiv_id":"...", "section":"..."},
    ...
  ],
  "passages": [
    {"text":"...", "arxiv_id":"...", "section":"...", "source_html":"...", "score":0.42},
    ...
  ],
  "followups": []
 }
```
If you get insufficient_quota or auth errors here, the embeddings still work (ONNX is local) amd the LLM just requires a valid OPENAI_API_KEY in .env


