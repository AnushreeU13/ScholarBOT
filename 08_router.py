"""
08_router.py
LLM-based domain and intent router — replaces the old keyword-list router.
Architecture position: 08 — runs after context resolution, before retrieval.

Returns a structured RouteDecision dict from a single LLM call (JSON output).
Covers domain classification, intent detection, KB selection, and abstain signal
in one place — no keyword lists, no brittle string matching.

RouteDecision keys:
  domain      : "TB" | "pneumonia" | "drug" | "out_of_domain"
  intent      : "definition" | "diagnosis" | "treatment" | "prevention"
                | "drug_info" | "summarize" | "general"
  target_kbs  : list of KB_* constants (may be empty if abstain)
  abstain     : bool — True means do not retrieve, return abstain response
  reason      : short explanation (for logging)
"""

from __future__ import annotations

import importlib
import json
import os
import re
from typing import Dict, List

_cfg = importlib.import_module("01_config")

# ── Known TB/pneumonia drugs (for mixed-intent detection) ─────────────────────
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


# ── JSON parser ───────────────────────────────────────────────────────────────

def _parse_json(text: str) -> Dict:
    """Extract JSON object from LLM output robustly."""
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


# ── LLM call ──────────────────────────────────────────────────────────────────

def _llm_route(query: str, has_user_doc: bool) -> Dict:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        # No API key — safe fallback: abstain
        return {
            "domain": "out_of_domain", "intent": "general",
            "target_kbs": [], "abstain": True,
            "reason": "No OpenAI API key set.",
        }

    try:
        from openai import OpenAI
        client = OpenAI()
        prompt = _ROUTER_PROMPT.format(
            query=query,
            has_user_doc="yes" if has_user_doc else "no",
        )
        resp = client.chat.completions.create(
            model=_cfg.ROUTER_MODEL,
            messages=[
                {"role": "system", "content": _ROUTER_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,
            max_tokens=_cfg.MAX_TOKENS_ROUTER,
        )
        raw = (resp.choices[0].message.content or "").strip()
        result = _parse_json(raw)
        if result:
            return result
    except Exception as e:
        print(f"[Router] LLM call failed: {e} — using keyword fallback.")

    # ── Keyword fallback (minimal, for when LLM is unavailable) ───────────────
    return _keyword_fallback(query, has_user_doc)


def _keyword_fallback(query: str, has_user_doc: bool) -> Dict:
    """
    Minimal keyword-based fallback used only when the LLM is unreachable.
    Less accurate than LLM routing but maintains fail-closed behaviour.
    """
    q = query.lower()

    has_tb   = any(t in q for t in ["tuberculosis", " tb ", "tb.", "latent tb"])
    has_tb   = has_tb or bool(re.search(r"\btb\b", q))
    has_cap  = any(t in q for t in ["pneumonia", "cap", "community acquired"])
    has_drug = any(d in q for d in _KNOWN_DRUGS) or any(
        t in q for t in ["drug", "medication", "dosage", "side effect", "adverse", "interaction"]
    )

    if not has_tb and not has_cap and not has_drug:
        return {"domain": "out_of_domain", "intent": "general",
                "target_kbs": [], "abstain": True, "reason": "No domain match."}

    target_kbs: List[str] = []
    domain = "TB" if has_tb else ("pneumonia" if has_cap else "drug")

    if has_drug:
        target_kbs.append(_cfg.KB_DRUGLABELS)
    if has_tb or has_cap:
        target_kbs.append(_cfg.KB_GUIDELINES)

    intent = "general"
    if "summar" in q or "overview" in q:
        intent = "summarize"
        if has_user_doc:
            target_kbs = [_cfg.KB_USER]
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


# ── Public API ────────────────────────────────────────────────────────────────

def route(query: str, has_user_doc: bool = False, force_user_kb: bool = False) -> Dict:
    """
    Main entry point.

    Args:
        query        : The resolved (coreference-clean) query string.
        has_user_doc : Whether the user has uploaded a document this session.
        force_user_kb: If True (User Document Only mode), lock target_kbs to [KB_USER].

    Returns a RouteDecision dict.
    """
    if force_user_kb:
        # Bypass LLM — we already know the target
        return {
            "domain":     "user_doc",
            "intent":     "general",
            "target_kbs": [_cfg.KB_USER],
            "abstain":    False,
            "reason":     "Force user_kb mode.",
        }

    decision = _llm_route(query, has_user_doc)

    # Normalise target_kbs to use config constants
    _KB_MAP = {
        "guidelines_kb": _cfg.KB_GUIDELINES,
        "druglabels_kb": _cfg.KB_DRUGLABELS,
        "user_kb":       _cfg.KB_USER,
    }
    decision["target_kbs"] = [
        _KB_MAP.get(k, k) for k in decision.get("target_kbs", [])
    ]

    print(f"[Router] domain={decision.get('domain')} intent={decision.get('intent')} "
          f"kbs={decision.get('target_kbs')} abstain={decision.get('abstain')} "
          f"reason={decision.get('reason')}")

    return decision
