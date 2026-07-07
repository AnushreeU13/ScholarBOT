"""Single shared entry point for OpenAI chat completions and JSON parsing."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict

logger = logging.getLogger("scholarbot.llm")


def complete(system: str, prompt: str, max_tokens: int, model: str, temperature: float = 0) -> str:
    """
    Returns the raw text response, or "" if no API key / on any error.
    Fails open by design (callers abstain rather than crash on an LLM outage),
    but the failure is logged so it's visible in server logs instead of
    silently degrading every request to "abstain" with no diagnostic trail.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning("OPENAI_API_KEY is not set — LLM calls will no-op and callers will abstain.")
        return ""
    try:
        from openai import OpenAI

        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.error("OpenAI call failed (model=%s): %s: %s", model, type(e).__name__, e)
        return ""


def parse_json(text: str) -> Dict:
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return {}
