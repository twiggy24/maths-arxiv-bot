# Turn retrieved passages into short, citable "claims" WITHOUT an LLM.
from typing import List, Dict
import re

def _first_sentences(text: str, max_sents: int = 2) -> List[str]:
    """
    Sentence splitter: split on . ? !
    Returns up to max_sents non-empty fragments.
    """
    parts = re.split(r"(?<=[\.\?\!])\s+", text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts[:max_sents] or ([text.strip()] if text.strip() else [])

async def extract_claims(question: str, passages: List[Dict], max_claims: int = 8) -> List[Dict]:
    """
    Build atomic, citable claims from the top passages:
      - take the first 1–2 sentences from each passage
      - trim & dedupe
    Output: [{"claim","arxiv_id","section"}]
    """
    claims: List[Dict] = []
    seen = set()

    for p in passages:
        for sent in _first_sentences(p.get("text", ""), max_sents=2):
            claim = sent.strip()
            if not claim:
                continue
            key = (claim, p.get("arxiv_id", ""), p.get("section", ""))
            if key in seen:
                continue
            seen.add(key)
            claims.append(
                {
                    "claim": claim,
                    "arxiv_id": p.get("arxiv_id", ""),
                    "section": p.get("section", ""),
                }
            )
            if len(claims) >= max_claims:
                return claims
    return claims



