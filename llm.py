# llm.py
# Minimal async wrappers around the OpenAI Chat API.
# Safe when USE_LLM=false (no network calls).

import os
import json
from typing import List, Dict, Optional, Any
from settings import settings

USE_LLM = os.getenv("USE_LLM", "false").lower() == "true"

if USE_LLM:
    # Only import the SDK when we actually plan to use it
    from openai import OpenAI
    _client = OpenAI(api_key=settings.OPENAI_API_KEY)

async def chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Return a single assistant message string.
    """
    if not USE_LLM:
        raise RuntimeError("chat() called but USE_LLM=false. Set USE_LLM=true in .env to enable LLM calls.")

    resp = _client.chat.completions.create(
        model=model or settings.LLM_MODEL,
        messages=messages,
        temperature=settings.LLM_TEMPERATURE if temperature is None else temperature,
        max_tokens=settings.LLM_MAX_TOKENS if max_tokens is None else max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()

async def chat_json(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Any:
    """
    Ask the model to return JSON (array or object). We parse it into Python.
    """
    if not USE_LLM:
        #Upstream code should have a fallback; return an empty structure.
        return []

    resp = _client.chat.completions.create(
        model=model or settings.LLM_MODEL,
        messages=messages,
        temperature=settings.LLM_TEMPERATURE if temperature is None else temperature,
        max_tokens=settings.LLM_MAX_TOKENS if max_tokens is None else max_tokens,
        # could add response_format if you later want hard JSON enforcement.
    )
    raw = resp.choices[0].message.content or ""
    try:
        return json.loads(raw)
    except Exception:
        # Try to salvage a JSON array embedded in text
        start = raw.find('[')
        end = raw.rfind(']')
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start:end + 1])
        # Or a JSON object
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start:end + 1])
        return []




    