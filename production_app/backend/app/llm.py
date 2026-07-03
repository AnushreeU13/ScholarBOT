"""Single shared entry point for OpenAI chat completions and JSON parsing."""

from __future__ import annotations

import json
import os
import re
from typing import Dict


def complete(system: str, prompt: str, max_tokens: int, model: str, temperature: float = 0) -> str:
    """Returns the raw text response, or "" if no API key / on any error."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
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
    except Exception:
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
