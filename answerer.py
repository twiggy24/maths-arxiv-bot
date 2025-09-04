# Compose final answer from claims.
# If USE_LLM=false (default), return a bullet list with citations (no OpenAI needed).
# If USE_LLM=true, call the LLM via llm.chat(...) and prompts.ANSWER_COMPOSER.

from typing import List, Dict
import os

USE_LLM = os.getenv("USE_LLM", "false").lower() == "true"

if USE_LLM:
    from llm import chat
    from prompts import ANSWER_COMPOSER

async def compose_answer(question: str, claims: List[Dict]) -> str:
    if not claims:
        return "I couldn't find grounded passages for that question."

    # Prepare consistent citation tags
    bullets = []
    for c in claims:
        tag = f"[arXiv:{c.get('arxiv_id','')}, {c.get('section','Section')}]"
        bullets.append(f"- {c['claim']} {tag}")

    if not USE_LLM:
        header = f"Grounded statements related to: {question}\n"
        return header + "\n".join(bullets)

    # LLM path
    messages = [
        {"role": "system", "content": ANSWER_COMPOSER},
        {"role": "user", "content": f"Question: {question}\n\nClaims:\n" + "\n".join(bullets)},
    ]
    return await chat(messages)




