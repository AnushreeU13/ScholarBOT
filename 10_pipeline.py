"""
10_pipeline.py
RAG pipeline with separate execution paths by intent.
Architecture position: 10 — receives route + retriever, produces PipelineResult.

Intent paths:
  out_of_domain / abstain  → immediate abstain, no retrieval
  summarize                → stratified sampling + summarize prompt
  drug_info                → retrieval from druglabels_kb + drug-specific prompt
  definition/diagnosis/
  treatment/prevention/
  general                  → retrieval from guidelines_kb (+ druglabels if mixed)
                             + evidence-only QA prompt + patient rewrite

Key improvements vs old rag_pipeline_aligned.py:
- Structured JSON output from LLM (no regex bullet parsing)
- Chunk IDs in bullets = direct, unambiguous citations (no Jaccard alignment)
- Summarize is a first-class path with stratified sampling
- Self-critique loop uses strict evidence-only rule (no "foundational definitions" exception)
- All generation is evidence-only — supplementation with outside knowledge is forbidden
"""

from __future__ import annotations

import importlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_cfg = importlib.import_module("01_config")


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    status:          str          # "answer" | "abstain"
    clinician_bullets: List[str]  # evidence-only bullet points with [chunk_id] refs
    patient_bullets:   List[str]  # plain-language bullets
    citations:         List[str]  # deduplicated citation strings
    confidence:        float      # best reranker score (0–1)
    evidence_chunks:   List[Dict] # raw chunks for UI expander
    route:             Dict       # router decision (for debug)
    abstain_reason:    str = ""


# ── LLM helper ────────────────────────────────────────────────────────────────

def _llm(system: str, prompt: str, max_tokens: int) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return json.dumps({"status": "abstain", "clinician_bullets": [], "patient_bullets": []})
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=_cfg.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[Pipeline] LLM error: {e}")
        return ""


def _parse_json(text: str) -> Dict:
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


# ── Evidence block builder ────────────────────────────────────────────────────

def _build_evidence_block(chunks: List[Dict]) -> str:
    """Format chunks as a numbered evidence block for the LLM prompt."""
    lines = []
    for c in chunks:
        lines.append(f"[{c['chunk_id']}] {c['text']}")
    return "\n\n".join(lines)


def _collect_citations(chunks: List[Dict]) -> List[str]:
    seen, out = set(), []
    for c in chunks:
        cit = c.get("citation", "Unknown source")
        if cit not in seen:
            seen.add(cit)
            out.append(cit)
    return out


# ── Prompt templates ──────────────────────────────────────────────────────────

_QA_SYSTEM = (
    "You are a strict clinical evidence synthesizer. "
    "You ONLY use the provided evidence. "
    "You NEVER add information not present in the evidence. "
    "If evidence is insufficient, you output ABSTAIN. "
    "Output ONLY valid JSON — no markdown, no explanation."
)

_QA_PROMPT = """Answer the QUESTION using ONLY the EVIDENCE below.
Every bullet must cite its source chunk number as [N].
Do NOT add any information not explicitly stated in the evidence.
If the evidence does not contain enough to answer, output the abstain JSON.

QUESTION: {query}

EVIDENCE:
{evidence}

Output JSON:
{{
  "status": "answer",
  "clinician_bullets": [
    "Claim sentence grounded in evidence. [1]",
    "Another claim. [2]"
  ],
  "patient_bullets": [
    "Plain-language version of the same information. [1]"
  ]
}}

If evidence is insufficient:
{{"status": "abstain", "clinician_bullets": [], "patient_bullets": []}}"""


_DRUG_SYSTEM = (
    "You are a strict clinical pharmacist synthesizer. "
    "Extract ONLY what is explicitly stated in the drug label evidence. "
    "Do not add mechanism explanations or background information. "
    "Output ONLY valid JSON."
)

_DRUG_PROMPT = """Answer the QUESTION using ONLY the DRUG LABEL EVIDENCE below.
Extract only items explicitly present. Cite each point with [N].

QUESTION: {query}

EVIDENCE:
{evidence}

Output JSON:
{{
  "status": "answer",
  "clinician_bullets": [
    "Extracted drug label fact. [1]"
  ],
  "patient_bullets": [
    "Plain-language explanation. [1]"
  ]
}}

If evidence is insufficient:
{{"status": "abstain", "clinician_bullets": [], "patient_bullets": []}}"""


_SUMMARIZE_SYSTEM = (
    "You are a clinical document summarizer. "
    "Summarize ONLY what is present in the provided document excerpts. "
    "Do not add outside knowledge. Output ONLY valid JSON."
)

_SUMMARIZE_PROMPT = """Summarize the following document excerpts into a structured overview.
Cite each point with [N] (the excerpt number it came from).

DOCUMENT EXCERPTS:
{evidence}

Output JSON:
{{
  "status": "answer",
  "clinician_bullets": [
    "Key clinical finding or section summary. [1]",
    "Another key point. [2]"
  ],
  "patient_bullets": [
    "Plain-language summary point. [1]"
  ]
}}"""


_CRITIQUE_SYSTEM = (
    "You are a clinical peer-reviewer checking for hallucinations. "
    "Output ONLY valid JSON."
)

_CRITIQUE_PROMPT = """Review the DRAFT ANSWER against the EVIDENCE.
Remove any bullet that is NOT explicitly supported by the evidence.
If a bullet cites [N], verify the claim appears in chunk [N].
Keep only well-supported bullets.

DRAFT ANSWER:
{draft}

EVIDENCE:
{evidence}

Output JSON with only the surviving bullets:
{{
  "clinician_bullets": ["surviving bullet. [1]"],
  "patient_bullets":   ["surviving patient bullet. [1]"]
}}

If nothing survives: {{"clinician_bullets": [], "patient_bullets": []}}"""


# ── RAGPipeline ───────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Orchestrates retrieval → generation → critique → patient rewrite.
    Separate execution paths for QA, drug info, and summarization.
    """

    def __init__(self, retriever):
        self.retriever = retriever   # HybridRetriever from 09_retriever.py

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, query: str, route: Dict) -> PipelineResult:
        if route.get("abstain"):
            return self._abstain(route.get("reason", "out_of_scope"))

        intent     = route.get("intent", "general")
        target_kbs = route.get("target_kbs", [])

        if intent == "summarize":
            return self._summarize(query, target_kbs)
        elif route.get("domain") == "drug" or intent == "drug_info":
            return self._drug_qa(query, target_kbs)
        else:
            return self._guideline_qa(query, target_kbs, intent)

    # ── Abstain ───────────────────────────────────────────────────────────────

    @staticmethod
    def _abstain(reason: str) -> PipelineResult:
        return PipelineResult(
            status="abstain",
            clinician_bullets=[],
            patient_bullets=[],
            citations=[],
            confidence=0.0,
            evidence_chunks=[],
            route={},
            abstain_reason=reason,
        )

    # ── Guideline QA ──────────────────────────────────────────────────────────

    def _guideline_qa(self, query: str, target_kbs: List[str], intent: str) -> PipelineResult:
        chunks = self.retriever.retrieve(query, target_kbs)
        if not chunks:
            return self._abstain("no_chunks_retrieved")

        # Confidence gate
        best_score = chunks[0]["score"]
        threshold  = max(
            _cfg.KB_SIM_THRESHOLD.get(kb, _cfg.DEFAULT_SIM_THRESHOLD)
            for kb in target_kbs
        )
        if best_score < threshold:
            return self._abstain(f"low_confidence ({best_score:.3f} < {threshold})")

        # Evidence sufficiency check
        from importlib import import_module
        _ret = import_module("09_retriever")
        if not _ret._check_sufficiency(query, chunks):
            return self._abstain("evidence_insufficient_for_query")

        evidence = _build_evidence_block(chunks)
        raw      = _llm(_QA_SYSTEM, _QA_PROMPT.format(query=query, evidence=evidence),
                        _cfg.MAX_TOKENS_ANSWER)
        parsed   = _parse_json(raw)

        if not parsed or parsed.get("status") == "abstain" or not parsed.get("clinician_bullets"):
            return self._abstain("llm_abstain")

        # Self-critique
        parsed = self._critique(parsed, evidence)
        if not parsed.get("clinician_bullets"):
            return self._abstain("critique_rejected_all")

        return PipelineResult(
            status="answer",
            clinician_bullets=parsed["clinician_bullets"],
            patient_bullets=parsed.get("patient_bullets", []),
            citations=_collect_citations(chunks),
            confidence=best_score,
            evidence_chunks=[{
                "chunk_id": c["chunk_id"], "text": c["text"][:1200],
                "citation": c["citation"], "store": c["store"],
            } for c in chunks],
            route={},
        )

    # ── Drug QA ───────────────────────────────────────────────────────────────

    def _drug_qa(self, query: str, target_kbs: List[str]) -> PipelineResult:
        # Ensure druglabels_kb is included
        kbs = target_kbs if target_kbs else [_cfg.KB_DRUGLABELS]
        if _cfg.KB_DRUGLABELS not in kbs:
            kbs = [_cfg.KB_DRUGLABELS] + kbs

        chunks = self.retriever.retrieve(query, kbs)
        if not chunks:
            return self._abstain("no_drug_chunks_retrieved")

        best_score = chunks[0]["score"]
        if best_score < _cfg.KB_SIM_THRESHOLD.get(_cfg.KB_DRUGLABELS, 0.5):
            return self._abstain(f"low_confidence ({best_score:.3f})")

        evidence = _build_evidence_block(chunks)
        raw      = _llm(_DRUG_SYSTEM, _DRUG_PROMPT.format(query=query, evidence=evidence),
                        _cfg.MAX_TOKENS_ANSWER)
        parsed   = _parse_json(raw)

        if not parsed or parsed.get("status") == "abstain" or not parsed.get("clinician_bullets"):
            return self._abstain("llm_abstain")

        return PipelineResult(
            status="answer",
            clinician_bullets=parsed["clinician_bullets"],
            patient_bullets=parsed.get("patient_bullets", []),
            citations=_collect_citations(chunks),
            confidence=best_score,
            evidence_chunks=[{
                "chunk_id": c["chunk_id"], "text": c["text"][:1200],
                "citation": c["citation"], "store": c["store"],
            } for c in chunks],
            route={},
        )

    # ── Summarize ─────────────────────────────────────────────────────────────

    def _summarize(self, query: str, target_kbs: List[str]) -> PipelineResult:
        # Use user_kb if available, otherwise first target
        store_name = _cfg.KB_USER if _cfg.KB_USER in target_kbs else (target_kbs[0] if target_kbs else None)
        if not store_name:
            return self._abstain("no_target_kb_for_summarize")

        chunks = self.retriever.stratified_sample(store_name, n=16)
        if not chunks:
            return self._abstain("empty_store_for_summarize")

        evidence = _build_evidence_block(chunks)
        raw      = _llm(_SUMMARIZE_SYSTEM,
                        _SUMMARIZE_PROMPT.format(evidence=evidence),
                        _cfg.MAX_TOKENS_SUMMARIZE)
        parsed   = _parse_json(raw)

        if not parsed or not parsed.get("clinician_bullets"):
            return self._abstain("summarize_llm_empty")

        return PipelineResult(
            status="answer",
            clinician_bullets=parsed["clinician_bullets"],
            patient_bullets=parsed.get("patient_bullets", []),
            citations=_collect_citations(chunks),
            confidence=1.0,   # not score-gated for summarization
            evidence_chunks=[{
                "chunk_id": c["chunk_id"], "text": c["text"][:1200],
                "citation": c["citation"], "store": c["store"],
            } for c in chunks],
            route={},
        )

    # ── Self-critique ─────────────────────────────────────────────────────────

    def _critique(self, parsed: Dict, evidence: str) -> Dict:
        """
        Prune any bullet not explicitly supported by the evidence.
        Strict: no outside knowledge is allowed to survive.
        """
        draft = json.dumps({
            "clinician_bullets": parsed.get("clinician_bullets", []),
            "patient_bullets":   parsed.get("patient_bullets",   []),
        }, indent=2)

        raw    = _llm(_CRITIQUE_SYSTEM,
                      _CRITIQUE_PROMPT.format(draft=draft, evidence=evidence),
                      _cfg.MAX_TOKENS_ANSWER)
        result = _parse_json(raw)

        # Fall back to original if critique produced nothing parseable
        if not result:
            return parsed
        return result
