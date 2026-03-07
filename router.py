"""
router.py

Routing for Tier-1 clinical RAG.

Design goals:
- Guidelines KB ONLY for TB and pneumonia/CAP (+ guideline-signal queries).
- Druglabels KB for medication questions (DailyMed SPL).
- Fail-closed by default.

Key fix:
- preferred_section_groups is ONLY for section grouping (adverse, interactions, dosage, warnings, etc.)
- task_hints is for "diagnosis / treatment / prevention" style tasks, not used for section bias directly.
"""

from __future__ import annotations
from dataclasses import dataclass
from dataclasses import dataclass
from typing import List
import re


@dataclass
class RouteDecision:
    intent: str
    target_kbs: List[str]
    preferred_section_groups: List[str]
    reason: str
    task_hints: List[str]  # NEW (safe): does not break callers that ignore it


def _normalize(q: str) -> str:
    return (q or "").strip().lower()


_GUIDE_DOMAIN_TRIGGERS = [
    "tuberculosis", " tb ", "tb.", "tb,", "latent tb", "active tb",
    "pneumonia", "cap", "community acquired pneumonia", "community-acquired pneumonia",
    "ats", "idsa", "who", "cdc", "guideline", "recommendation", "recommendations",
]

_OUT_OF_DOMAIN_TRIGGERS = [
    "diabetes", "heart attack", "myocardial infarction", "stroke",
    "hypertension", "asthma", "cancer", "kidney failure", "seizure",
]

_DRUG_TRIGGERS = [
    "isoniazid", "rifampin", "rifampicin", "pyrazinamide", "ethambutol",
    "drug", "medication", "tablet", "capsule", "mg", "dose", "dosage",
    "adverse", "side effect", "reaction", "toxicity", "contraindication",
    "interaction", "boxed warning", "precaution",
]

# Task hints (NOT section groups)
_TASK_DIAGNOSIS = ["diagnose", "diagnosed", "diagnosis", "testing", "test", "suspect", "screen", "workup"]
_TASK_TREATMENT = ["treat", "treatment", "therapy", "management", "antibiotic", "regimen"]
_TASK_PREVENTION = ["prevention", "prevent", "prophylaxis", "vaccin"]


def route_query(query: str, user_uploaded_available: bool = False) -> RouteDecision:
    q = _normalize(query)

    preferred: List[str] = []
    task_hints: List[str] = []

    # --- preferred section groups (clean) ---
    if any(k in q for k in ["dose", "dosing", "dosage", "mg", "administration"]):
        preferred.append("dosage")

    if any(k in q for k in ["boxed", "boxed warning", "warning", "warnings", "precaution", "black box"]):
        preferred.append("warnings")

    if any(k in q for k in ["contraindication", "contraindications", "do not use", "should not be used"]):
        preferred.append("contraindications")

    if any(k in q for k in [
        "pregnan", "lactat", "breastfeed",
        "renal", "kidney", "hepatic", "liver",
        "pediatric", "children", "child", "neonate",
        "geriatric", "elderly", "older adult",
        "specific population", "populations"
    ]):
        preferred.append("populations")

    if any(k in q for k in ["adverse", "side effect", "reaction", "toxicity"]):
        preferred.append("adverse")
    if any(k in q for k in ["interaction", "cyp", "inhibitor", "inducer"]):
        preferred.append("interactions")
    if any(k in q for k in ["indication", "indications", "used for", "approved for"]):
        preferred.append("indications")

    # --- task hints (separate) ---
    if any(t in q for t in _TASK_DIAGNOSIS):
        task_hints.append("diagnosis")
    if any(t in q for t in _TASK_TREATMENT):
        task_hints.append("treatment")
    if any(t in q for t in _TASK_PREVENTION):
        task_hints.append("prevention")

    # Regex for robust boundary detection (fixes "TB?" or "TB!")
    has_guide_domain = any(t in q for t in _GUIDE_DOMAIN_TRIGGERS)
    if not has_guide_domain:
        # Fallback to regex for short acronyms
        if re.search(r"\b(tb|cap)\b", q):
            has_guide_domain = True
    has_drug = any(t in q for t in _DRUG_TRIGGERS)
    has_ood = any(t in q for t in _OUT_OF_DOMAIN_TRIGGERS)

    # --- guideline-oriented preferences (separate namespace) ---
    # These do NOT affect drug labels; they help bias guideline retrieval.
    if has_guide_domain:
        if "diagnosis" in task_hints:
            preferred.append("g_diagnosis")
        if "treatment" in task_hints:
            preferred.append("g_treatment")
        if "prevention" in task_hints:
            preferred.append("g_prevention")

    targets: List[str] = []
    if user_uploaded_available:
        targets.append("user_fact_kb_medcpt")

    # Out-of-domain guard
    if has_ood and not has_guide_domain:
        if has_drug:
            targets.append("kb_druglabels_medcpt")
            return RouteDecision("drug", targets, preferred, "Out-of-domain for guidelines; route to druglabels only.", task_hints)
        return RouteDecision("abstain", targets, preferred, "Out-of-domain and not a druglabel question.", task_hints)

    # Guideline domain
    if has_guide_domain and (not has_drug):
        targets.append("kb_guidelines_medcpt")
        return RouteDecision("guideline", targets, preferred, "Matched TB/CAP guideline domain.", task_hints)

    # Druglabel domain
    if has_drug and not has_guide_domain:
        targets.append("kb_druglabels_medcpt")
        return RouteDecision("drug", targets, preferred, "Matched druglabel triggers.", task_hints)

    # Mixed allowed only when within TB/CAP domain
    if has_guide_domain:
        targets.extend(["kb_druglabels_medcpt", "kb_guidelines_medcpt"])
        return RouteDecision("mixed", targets, preferred, "Mixed intent within TB/CAP domain; query both.", task_hints)

    return RouteDecision("abstain", targets, preferred, "No clear domain match (fail-closed).", task_hints)
