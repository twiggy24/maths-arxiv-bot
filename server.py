# server.py
# FastAPI app wiring together retrieval -> claims -> answer.

from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from retrieval import retrieve_passages          # async
from claims import extract_claims                # async
from answerer import compose_answer              # async
from reflect import reflect_two_hop              # async, no-ops when USE_LLM=false

app = FastAPI(title="Math ArXiv Bot", version="0.1")

class AskPayload(BaseModel):
    question: str
    top_k: int = 8

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/ask")
async def ask(payload: AskPayload):
    """
    1) embed & search (Qdrant)
    2) turn top passages into short claims (no-LLM baseline)
    3) (optional) reflect to produce follow-up queries (no-op in local mode)
    4) compose final answer (no-LLM baseline unless USE_LLM=true)
    """
    try:
        # 1) retrieval
        passages: List[Dict] = await retrieve_passages(payload.question, limit=payload.top_k)

        # 2) claims
        claims: List[Dict] = await extract_claims(payload.question, passages)

        # 3) (optional) reflection
        followups: List[str] = await reflect_two_hop(payload.question, claims)

        # 4) compose final answer
        answer: str = await compose_answer(payload.question, claims)

        return {
            "answer": answer,
            "claims": claims,
            "passages": passages,
            "followups": followups,
        }
    except Exception as e:
        # Print traceback to the Uvicorn console and return a 500 with detail
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


    


