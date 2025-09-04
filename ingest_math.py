"""Download arXiv math.AG + math.NT, parse, chunk, embed, and upsert into Qdrant."""
import time
import uuid 
import requests
from typing import List
from qdrant_client.models import PointStruct 
from settings import settings
from db_qdrant import connect, ensure_collection
from embedder import embed_texts
from html_parse import fetch_ar5iv_html, html_to_sections, chunk_text, AR5IV_BASE
import re

BAD_SNIPPETS = (
    "View a PDF",          # ar5iv header
    "HTML (experimental)", # ar5iv header
    "Access Paper",        # ar5iv header
    "BibTeX",              # citation widget
    "×",                   # close button glyph
)

def looks_junky(text: str) -> bool:
    """True if the paragraph/snippet is clearly not paper content."""
    t = (text or "").strip()
    if not t:
        return True
    #Drop very short UI strings
    if len(t) < 50:
        return True
    # Drop known ar5iv UI bits
    if any(b in t for b in BAD_SNIPPETS):
        return True
    # Mostly punctuation / symbols? (quite defensive)
    letters = sum(ch.isalpha() for ch in t)
    if letters < 0.3 * len(t):
        return True
    return False

def clean_whitespace(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "")
    return s.strip()


ARXIV_API = "http://export.arxiv.org/api/query"


def list_recent_arxiv_ids(max_results: int = 50) -> List[str]:
    """Use the arXiv Atom API to get recent ids for math.AG and math.NT."""
    url = (
        f"{ARXIV_API}?search_query=cat:math.AG+OR+cat:math.NT&"
        "sortBy=submittedDate&sortOrder=descending&start=0&"
        f"max_results={max_results}"
    )   
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    ids = []
    for line in r.text.splitlines():
        if "<id>http://arxiv.org/abs" in line:
            aid = line.split("/abs/")[-1].split("</id>")[0]
            ids.append(aid)
    # de-duplicate while preserving order
    seen = set()
    out = []
    for a in ids:
        if a not in seen:
            out.append(a)
            seen.add(a)
    return out


def run(max_results: int = 50, batch_upsert: int = 128):
    client = connect()
    ensure_collection(client) # BGE-M3 dense size

    for aid in list_recent_arxiv_ids(max_results=max_results):
        try:
            html = fetch_ar5iv_html(aid)
            sections = html_to_sections(html)

            texts, metas = [], []
            for (sect_title, sect_text) in sections:
                for chunk in chunk_text(sect_text, 200, 40):  # 200w with ~40w overlap
                    # Clean and filter BEFORE indexing
                    text = clean_whitespace(chunk)
                    if looks_junky(text):
                        continue  #skip header/widgets like "View a PDF...", "BibTeX", "×", etc.

                    texts.append(text)
                    metas.append({
                        "arxiv_id": aid,
                        "section": sect_title,
                        "source_html": AR5IV_BASE + aid,
                        # Store under 'text' so the retriever can read it uniformly
                        "text": text,
                    })


            if not texts:
                print(f"No text chunks for {aid}; skipping")
                continue

            vecs = embed_texts(texts)
            points = []
            for i in range(len(texts)):
                points.append(PointStruct(id=str(uuid.uuid4()), vector=vecs[i], payload=metas[i]))

            # Upsert in batches for large papers
            for i in range(0, len(points), batch_upsert):
                client.upsert(collection_name=settings.COLLECTION_NAME, points=points[i:i + batch_upsert])
            print(f"Indexed {aid} with {len(points)} chunks")
            time.sleep(0.4) # delay
        except Exception as e:
            print("Skip", aid, "->", e)

if __name__ == "__main__":
    run(max_results=30)


