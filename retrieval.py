from typing import List, Dict
from settings import settings
from db_qdrant import connect
from embedder import embed_queries  # <- use the query encoder (adds "query: " prefix)

async def retrieve_passages(query: str, limit: int = 20) -> List[Dict]:
    """
    Embed query, search Qdrant, return normalised passages dicts with text.
    Each item: {"text","arxiv_id","section","source_html","score"}.
    """
    client = connect()

    # 1) Embed the query as a single vector (BGE-M3 "query:" prefix applied inside)
    qv = embed_queries([query])[0]

    # 2) Vector search
    hits = client.search(
        collection_name=settings.COLLECTION_NAME,
        query_vector=qv,
        limit=limit,
        with_payload=True,
    )

    # 3) Normalise payloads
    out: List[Dict] = []
    for h in hits:
        p = h.payload or {}
        text = p.get("text") or p.get("chunk_text") or ""
        out.append({
            "score": h.score,
            "text": text,
            "arxiv_id": p.get("arxiv_id", ""),
            "section": p.get("section", ""),
            "source_html": p.get("source_html", ""),
        })
    return out




    