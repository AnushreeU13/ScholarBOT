"""
LLM-based domain and intent router.
Keyword pre-filter first (free, instant); LLM is only called when the
keyword pass finds no domain signal. Retrieval's confidence gate is the
final abstain backstop.
"""

from __future__ import annotations

import re
from typing import Dict, List

from app import config, llm

_TB_DRUGS = {
    "isoniazid", "rifampin", "rifampicin", "pyrazinamide", "ethambutol",
    "levofloxacin", "moxifloxacin", "linezolid", "bedaquiline", "delamanid",
    "streptomycin", "amikacin", "capreomycin", "cycloserine", "ethionamide",
}
_CAP_DRUGS = {
    "azithromycin", "amoxicillin", "doxycycline", "ceftriaxone",
    "levofloxacin", "moxifloxacin", "clarithromycin", "ampicillin",
}
_KNOWN_DRUGS = _TB_DRUGS | _CAP_DRUGS

_ROUTER_SYSTEM = (
    "You are a strict clinical query classifier for a medical RAG system "
    "that covers ONLY Tuberculosis (TB), Pneumonia (CAP), and their specific medications. "
    "Output ONLY valid JSON — no markdown, no explanation."
)

_ROUTER_PROMPT = """Classify the following clinical query.

Query: {query}
User has uploaded a document: {has_user_doc}

Rules:
- domain "TB": query is about tuberculosis or latent TB
- domain "pneumonia": query is about pneumonia or community-acquired pneumonia (CAP)
- domain "drug": query is specifically about a medication (dosage, side effects, interactions, warnings) — only TB/CAP drugs are in scope
- domain "out_of_domain": anything not about TB, pneumonia, or their medications (e.g. diabetes, cancer, HIV, heart disease)
- intent "summarize": user asks for a summary/overview of a document or topic
- intent "definition": user asks what something is
- intent "diagnosis": user asks how something is diagnosed or tested
- intent "treatment": user asks how something is treated or managed
- intent "prevention": user asks about prevention or prophylaxis
- intent "drug_info": drug-specific question (dosage, adverse effects, interactions, contraindications)
- intent "general": any other in-domain question
- target_kbs: use "guidelines_kb" for clinical/epidemiological questions, "druglabels_kb" for drug questions, both if mixed
- abstain: true ONLY if domain is "out_of_domain" OR if query is too vague to answer from evidence (e.g. "hello", "thanks")
- if has_user_doc is true and intent is "summarize", set target_kbs to ["user_kb"] only

Output JSON:
{{
  "domain": "TB" | "pneumonia" | "drug" | "out_of_domain",
  "intent": "definition" | "diagnosis" | "treatment" | "prevention" | "drug_info" | "summarize" | "general",
  "target_kbs": ["guidelines_kb"] | ["druglabels_kb"] | ["guidelines_kb", "druglabels_kb"] | ["user_kb"] | [],
  "abstain": false,
  "reason": "one sentence"
}}"""


def _llm_route(query: str, has_user_doc: bool) -> Dict:
    prompt = _ROUTER_PROMPT.format(query=query, has_user_doc="yes" if has_user_doc else "no")
    raw = llm.complete(_ROUTER_SYSTEM, prompt, config.MAX_TOKENS_ROUTER, config.ROUTER_MODEL)
    result = llm.parse_json(raw) if raw else {}
    if result:
        return result
    return _keyword_fallback(query, has_user_doc)


def _keyword_fallback(query: str, has_user_doc: bool) -> Dict:
    """Minimal keyword-based fallback used when the LLM is unavailable or unreachable."""
    q = query.lower()

    # A summarize request against the user's own uploaded document is
    # in-domain by virtue of the document, not the query wording — a natural
    # phrasing like "summarize the document I just uploaded" has no TB/
    # pneumonia/drug keyword and would otherwise fall through the domain gate
    # below and incorrectly abstain as out-of-domain. Check this first.
    if has_user_doc and ("summar" in q or "overview" in q):
        return {"domain": "user_doc", "intent": "summarize",
                "target_kbs": [config.COLLECTION_USER], "abstain": False,
                "reason": "Summarize request against uploaded document."}

    has_tb = any(t in q for t in ["tuberculosis", " tb ", "tb.", "latent tb"])
    has_tb = has_tb or bool(re.search(r"\btb\b", q))
    has_cap = any(t in q for t in ["pneumonia", "community acquired"]) or bool(re.search(r"\bcap\b", q))
    has_drug = any(d in q for d in _KNOWN_DRUGS) or any(
        t in q for t in ["drug", "medication", "dosage", "side effect", "adverse", "interaction"]
    )

    if not has_tb and not has_cap and not has_drug:
        return {"domain": "out_of_domain", "intent": "general",
                "target_kbs": [], "abstain": True, "reason": "No domain match."}

    target_kbs: List[str] = []
    domain = "TB" if has_tb else ("pneumonia" if has_cap else "drug")

    if has_drug:
        target_kbs.append(config.COLLECTION_DRUGLABELS)
    if has_tb or has_cap:
        target_kbs.append(config.COLLECTION_GUIDELINES)

    intent = "general"
    if "summar" in q or "overview" in q:
        intent = "summarize"
        if has_user_doc:
            target_kbs = [config.COLLECTION_USER]
    elif any(t in q for t in ["diagnos", "test", "workup"]):
        intent = "diagnosis"
    elif any(t in q for t in ["treat", "therap", "manag", "regimen"]):
        intent = "treatment"
    elif any(t in q for t in ["prevent", "prophylax"]):
        intent = "prevention"
    elif any(t in q for t in ["what is", "define", "definition"]):
        intent = "definition"
    elif has_drug:
        intent = "drug_info"

    return {"domain": domain, "intent": intent,
            "target_kbs": list(dict.fromkeys(target_kbs)),
            "abstain": False, "reason": "Keyword fallback."}


def route(query: str, has_user_doc: bool = False, force_user_kb: bool = False) -> Dict:
    """
    Fast path (~0 ms): clear keyword signal → route without any LLM call.
    Slow path (~800 ms): no keyword signal → LLM adjudicates.
    Abstain: only when BOTH keyword and LLM find no in-domain signal.
    """
    if force_user_kb:
        return {
            "domain": "user_doc",
            "intent": "general",
            "target_kbs": [config.COLLECTION_USER],
            "abstain": False,
            "reason": "Force user_kb mode.",
        }

    kw = _keyword_fallback(query, has_user_doc)
    if not kw["abstain"]:
        kw["reason"] = f"Keyword match (fast path). {kw.get('reason', '')}"
        return kw

    llm_decision = _llm_route(query, has_user_doc)
    if not llm_decision.get("abstain"):
        return llm_decision

    return kw
