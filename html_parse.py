"""Fetch ar5iv HTML and segment into sections and chunks suited for retrieval.
Heuristics preserve Theorem/Lemma/Definition blocks when possible.
"""
import re 
import requests 
from bs4 import BeautifulSoup

AR5IV_BASE = "https://ar5iv.org/html/"

HEADERS = {"User-Agent": "math-arxiv-bot/0.1 (contact: you@example.com)"}

MATH_MARK = " [MATH] "  # lightweight marker to avoid stripping math entirely


def fetch_ar5iv_html(arxiv_id: str) -> str:
    url = AR5IV_BASE + arxiv_id
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text


def html_to_sections(html: str):
    """Return list[(title, text)] extracting h2/h3 sections.
    Falls back to one big section if headers are missing. 
    Replaces <math> tags with a marker to keep tokenisation stable. 
    """
    soup = BeautifulSoup(html, "lxml")

    # Replace <math> blocks by a neutral marker so we don't lose context entirely.
    for eq in soup.find_all(["math"]):
        eq.replace_with(MATH_MARK)

    sections = []
    headers = soup.find_all(["h2", "h3"]) or []
    if headers:
        for h in headers: 
            title = h.get_text(" ", strip=True)
            # Collect sibling text until next header at the same level
            parts = []
            for sib in h.next_siblings:
                if getattr(sib, "name", None) in ["h2", "h3"]:
                    break
                parts.append(getattr(sib, "get_text", lambda *a, **k: str(sib))(" ", strip=True))
            text = re.sub(r"\s+", " ", " ".join(parts)).strip()
            if text:
                sections.append((title, text))
    else:
        body = soup.body.get_text(" ", strip=True) if soup.body else soup.get_text(" ", strip=True)
        body = re.sub(r"\s+", " ", body)
        sections = [("Body", body)]

    return sections 

def chunk_text(text: str, target_words: int = 200, overlap: int = 40):
    """Split text into ~target_words with token-friendly word overlap. 
    Overlap helps avoid cutting definitions/assumptions across chunks.
    """
    words = text.split()
    chunks = []
    i = 0
    step = max(1, target_words - overlap)
    while i < len(words):
        chunk = words[i:i + target_words]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        i += step
    return chunks


    