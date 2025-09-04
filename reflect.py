# reflect.py
# Optional "two-hop" query expansion. In local mode (USE_LLM=false) it no-ops.

import os
import json
from typing import List, Dict

USE_LLM = os.getenv("USE_LLM", "false").lower() == "true"

if USE_LLM:
    from llm import chat_json
    from prompts import REFLECT_2HOP

async def reflect_two_hop(question: str, claims: List[Dict]) -> List[str]:
    """
    Return up to 3 follow-up queries. If USE_LLM=false, returns [].
    """
    if not USE_LLM:
        return []

    messages = [
        {"role": "system", "content": REFLECT_2HOP},
        {"role": "user", "content": json.dumps({"question": question, "claims": claims})},
    ]
    data = await chat_json(messages)
    if isinstance(data, list):
        return [str(x) for x in data][:3]
    return []




