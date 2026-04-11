"""
07_context_manager.py
Conversation context management — first-class component, not a bolt-on.
Architecture position: 07 — runs before every query reaches the router.

Responsibilities:
1. Strip meta-references ("in the document I uploaded", "according to this file")
2. Detect pronoun / vague references and rewrite to self-contained clinical query
3. Maintain a rolling topic summary so coreference works across many turns
4. Detect topic shifts and reset the summary
"""

from __future__ import annotations

import importlib
import os
import re
from typing import List, Dict, Optional

_cfg = importlib.import_module("01_config")

# ── Patterns ──────────────────────────────────────────────────────────────────

# Meta-references the user makes to the upload itself — strip these
_META_REF = re.compile(
    r"\b(in (the |this )?(document|doc|file|paper|article|text|pdf|upload)"
    r"|according to (the |this )?(document|doc|file|paper|article|text|pdf|upload)"
    r"|based on (the |this )?(document|doc|file|paper|article|text|pdf|upload)"
    r"|from (the |this )?(document|doc|file|paper|article|text|pdf|upload)"
    r"|i (just )?uploaded"
    r"|i (just )?shared)\b",
    re.IGNORECASE,
)

# Pronouns and vague references that need resolution
_COREF_TRIGGERS = re.compile(
    r"\b(it|its|this|these|they|them|their|the condition|the disease|the infection"
    r"|the illness|the disorder|the treatment|the drug|the medication"
    r"|the pathogen|the bacteria|the organism)\b",
    re.IGNORECASE,
)

# Phrases that suggest a topic shift
_SHIFT_TRIGGERS = re.compile(
    r"\b(now (ask|talk|discuss)|switch(ing)? to|what about|tell me about"
    r"|change (the )?topic|different (question|topic))\b",
    re.IGNORECASE,
)


# ── LLM helper ────────────────────────────────────────────────────────────────

def _llm(prompt: str, max_tokens: int) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return ""
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=_cfg.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise clinical language assistant."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


# ── ContextManager ────────────────────────────────────────────────────────────

class ContextManager:
    """
    Maintains a rolling topic summary and rewrites ambiguous queries.
    One instance lives in the Streamlit session (via AlignedScholarBotEngine).
    Call reset() when the user clears chat.
    """

    def __init__(self):
        self.topic_summary: str = ""   # e.g. "tuberculosis (TB)"

    def reset(self) -> None:
        self.topic_summary = ""

    # ── Public entry point ────────────────────────────────────────────────────

    def resolve(self, raw_query: str, history: List[Dict]) -> str:
        """
        Returns a clean, self-contained clinical query ready for the router.
        Steps: strip meta-refs → detect shift → resolve coreference → return.
        """
        query = self._strip_meta_references(raw_query)

        # If user is clearly shifting topic, reset summary
        if _SHIFT_TRIGGERS.search(query):
            self.topic_summary = ""

        # Only call LLM for coreference if there's something to resolve
        if self.topic_summary and _COREF_TRIGGERS.search(query):
            query = self._resolve_coreference(query, history)

        return query

    def update_topic(self, resolved_query: str, answer_status: str) -> None:
        """
        After a successful answer, extract and store the clinical topic
        so future turns can resolve pronouns against it.
        Only updates on a real answer (not abstain).
        """
        if answer_status != "answer":
            return

        prompt = (
            f"Extract the primary clinical topic from this query in 1-4 words "
            f"(e.g. 'tuberculosis', 'isoniazid dosage', 'CAP treatment').\n"
            f"Query: {resolved_query}\n"
            f"Topic (1-4 words only):"
        )
        topic = _llm(prompt, max_tokens=_cfg.MAX_TOKENS_CONTEXT)
        if topic and len(topic) < 60:
            self.topic_summary = topic

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _strip_meta_references(query: str) -> str:
        """Remove phrases like 'in the document I uploaded' from the query."""
        cleaned = _META_REF.sub("", query)
        # Collapse leftover commas / whitespace at the start
        cleaned = re.sub(r"^[\s,]+", "", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        # If stripping removed almost everything, return the original
        if len(cleaned) < 5:
            return query
        return cleaned

    def _resolve_coreference(self, query: str, history: List[Dict]) -> str:
        """Use LLM to rewrite the query substituting pronouns with the known topic."""
        recent = history[-4:] if history else []
        history_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content'][:250]}"
            for m in recent
            if m.get("role") in ("user", "assistant") and m.get("content")
        )

        prompt = (
            f"Current clinical topic: {self.topic_summary}\n"
            f"Conversation history:\n{history_text}\n\n"
            f"Query: {query}\n\n"
            f"Rewrite the query replacing every pronoun or vague reference "
            f"with the specific clinical term it refers to. "
            f"Output only the rewritten query — no explanation, no quotes."
        )

        rewritten = _llm(prompt, max_tokens=_cfg.MAX_TOKENS_CONTEXT)
        if rewritten and 5 < len(rewritten) < 300:
            return rewritten
        return query
