"""
Conversation context management.
Strips meta-references, resolves pronouns against a rolling topic summary,
and resets on topic shift.
"""

from __future__ import annotations

import re
from typing import Dict, List

from app import config, llm

_META_REF = re.compile(
    r"\b(in (the |this )?(document|doc|file|paper|article|text|pdf|upload)"
    r"|according to (the |this )?(document|doc|file|paper|article|text|pdf|upload)"
    r"|based on (the |this )?(document|doc|file|paper|article|text|pdf|upload)"
    r"|from (the |this )?(document|doc|file|paper|article|text|pdf|upload)"
    r"|i (just )?uploaded"
    r"|i (just )?shared)\b",
    re.IGNORECASE,
)

_COREF_TRIGGERS = re.compile(
    r"\b(it|its|this|these|they|them|their|the condition|the disease|the infection"
    r"|the illness|the disorder|the treatment|the drug|the medication"
    r"|the pathogen|the bacteria|the organism)\b",
    re.IGNORECASE,
)

_SHIFT_TRIGGERS = re.compile(
    r"\b(now (ask|talk|discuss)|switch(ing)? to|what about|tell me about"
    r"|change (the )?topic|different (question|topic))\b",
    re.IGNORECASE,
)


class ContextManager:
    """Maintains a rolling topic summary and rewrites ambiguous queries."""

    def __init__(self):
        self.topic_summary: str = ""

    def reset(self) -> None:
        self.topic_summary = ""

    def resolve(self, raw_query: str, history: List[Dict]) -> str:
        query = self._strip_meta_references(raw_query)

        if _SHIFT_TRIGGERS.search(query):
            self.topic_summary = ""

        if self.topic_summary and _COREF_TRIGGERS.search(query):
            query = self._resolve_coreference(query, history)

        return query

    def update_topic(self, resolved_query: str, answer_status: str) -> None:
        if answer_status != "answer":
            return

        prompt = (
            f"Extract the primary clinical topic from this query in 1-4 words "
            f"(e.g. 'tuberculosis', 'isoniazid dosage', 'CAP treatment').\n"
            f"Query: {resolved_query}\n"
            f"Topic (1-4 words only):"
        )
        topic = llm.complete(
            "You are a precise clinical language assistant.",
            prompt,
            config.MAX_TOKENS_CONTEXT,
            config.OPENAI_MODEL,
        )
        if topic and len(topic) < 60:
            self.topic_summary = topic

    @staticmethod
    def _strip_meta_references(query: str) -> str:
        cleaned = _META_REF.sub("", query)
        cleaned = re.sub(r"^[\s,]+", "", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        if len(cleaned) < 5:
            return query
        return cleaned

    def _resolve_coreference(self, query: str, history: List[Dict]) -> str:
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

        rewritten = llm.complete(
            "You are a precise clinical language assistant.",
            prompt,
            config.MAX_TOKENS_CONTEXT,
            config.OPENAI_MODEL,
        )
        if rewritten and 5 < len(rewritten) < 300:
            return rewritten
        return query
