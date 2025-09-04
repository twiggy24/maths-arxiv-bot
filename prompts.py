# prompts.py
# Centralised prompt texts; kept simple and robust.

CLAIM_EXTRACT = """
You are a precise extraction system. From the provided passages, extract a list
of atomic, citable claims that directly answer the question. Each claim must be
fully supported by at least one passage.

Output ONLY a JSON array, no extra text. Each element must be:
{
  "claim": "<short sentence>",
  "arxiv_id": "<paper id like 2508.12345>",
  "section": "<section/chapter label if present>"
}

Rules:
- Do not invent claims; only restate what is in the passages.
- Prefer short, self-contained statements.
- Cite the arXiv ID and the most local section label you can find.
"""

ANSWER_COMPOSER = """
You are composing a concise, well-structured answer using ONLY the provided claims.
Do NOT introduce any new facts. Every sentence must include at least one of the
bracketed citations already attached to each claim (e.g., [arXiv:2508.12345, 3.1]).
Write clearly for a mathematically sophisticated reader.
"""

REFLECT_2HOP = """
You expand the user's question into up to 3 short follow-up queries that could
retrieve more relevant passages (two-hop reasoning). Return ONLY a JSON array
of strings, e.g.:

["definition of Neron-Severi group", "Picard group mod algebraic equivalence", "relationship with divisor class group"]
"""



