"""
RAG pipeline with separate execution paths by intent:
  abstain            → immediate abstain, no retrieval
  summarize          → stratified sampling + summarize prompt
  drug_info          → retrieval from druglabels_kb + drug-specific prompt
  everything else    → retrieval from guidelines_kb (+ druglabels if mixed)
                        + evidence-only QA prompt + self-critique

All generation is evidence-only — the LLM is instructed to abstain rather
than supplement with outside knowledge, and a self-critique pass prunes any
bullet not explicitly grounded in the retrieved evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from app import config, llm

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
  "clinician_bullets": ["Claim sentence grounded in evidence. [1]"],
  "patient_bullets": ["Plain-language version of the same information. [1]"]
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
  "clinician_bullets": ["Extracted drug label fact. [1]"],
  "patient_bullets": ["Plain-language explanation. [1]"]
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
  "clinician_bullets": ["Key clinical finding or section summary. [1]"],
  "patient_bullets": ["Plain-language summary point. [1]"]
}}"""

_CRITIQUE_SYSTEM = "You are a clinical peer-reviewer checking for hallucinations. Output ONLY valid JSON."

_CRITIQUE_PROMPT = """Review the DRAFT ANSWER against the EVIDENCE.
Remove any bullet that is NOT explicitly supported by the evidence.
If a bullet cites [N], verify the claim appears in chunk [N].
Keep only well-supported bullets.

DRAFT ANSWER:
{draft}

EVIDENCE:
{evidence}

Output JSON with only the surviving bullets:
{{"clinician_bullets": ["surviving bullet. [1]"], "patient_bullets": ["surviving patient bullet. [1]"]}}

If nothing survives: {{"clinician_bullets": [], "patient_bullets": []}}"""


@dataclass
class PipelineResult:
    status: str
    clinician_bullets: List[str]
    patient_bullets: List[str]
    citations: List[str]
    confidence: float
    evidence_chunks: List[Dict]
    route: Dict
    abstain_reason: str = ""


def _build_evidence_block(chunks: List[Dict]) -> str:
    return "\n\n".join(f"[{c['chunk_id']}] {c['text']}" for c in chunks)


def _collect_citations(chunks: List[Dict]) -> List[str]:
    seen, out = set(), []
    for c in chunks:
        cit = c.get("citation", "Unknown source")
        if cit not in seen:
            seen.add(cit)
            out.append(cit)
    return out


def _to_evidence_chunks(chunks: List[Dict]) -> List[Dict]:
    return [{
        "chunk_id": c["chunk_id"], "text": c["text"][:1200],
        "citation": c["citation"], "store": c["store"],
    } for c in chunks]


class RAGPipeline:
    """Orchestrates retrieval → generation → self-critique."""

    def __init__(self, retriever, sufficiency_check=None):
        self.retriever = retriever
        # Injectable so tests can bypass the LLM-backed sufficiency check.
        self._check_sufficiency = sufficiency_check
        if self._check_sufficiency is None:
            from app.retriever import check_sufficiency
            self._check_sufficiency = check_sufficiency

    def run(self, query: str, route: Dict) -> PipelineResult:
        if route.get("abstain"):
            return self._abstain(route.get("reason", "out_of_scope"))

        intent = route.get("intent", "general")
        target_kbs = route.get("target_kbs", [])

        if intent == "summarize":
            return self._summarize(target_kbs)
        elif route.get("domain") == "drug" or intent == "drug_info":
            return self._drug_qa(query, target_kbs)
        else:
            return self._guideline_qa(query, target_kbs)

    @staticmethod
    def _abstain(reason: str) -> PipelineResult:
        return PipelineResult(
            status="abstain", clinician_bullets=[], patient_bullets=[],
            citations=[], confidence=0.0, evidence_chunks=[], route={},
            abstain_reason=reason,
        )

    def _guideline_qa(self, query: str, target_kbs: List[str]) -> PipelineResult:
        chunks = self.retriever.retrieve(query, target_kbs)
        if not chunks:
            return self._abstain("no_chunks_retrieved")

        best_score = chunks[0]["score"]
        threshold = max(
            config.KB_SIM_THRESHOLD.get(kb, config.DEFAULT_SIM_THRESHOLD)
            for kb in target_kbs
        )
        if best_score < threshold:
            return self._abstain(f"low_confidence ({best_score:.3f} < {threshold})")

        if not self._check_sufficiency(query, chunks):
            return self._abstain("evidence_insufficient_for_query")

        evidence = _build_evidence_block(chunks)
        raw = llm.complete(_QA_SYSTEM, _QA_PROMPT.format(query=query, evidence=evidence),
                            config.MAX_TOKENS_ANSWER, config.OPENAI_MODEL)
        parsed = llm.parse_json(raw)

        if not parsed or parsed.get("status") == "abstain" or not parsed.get("clinician_bullets"):
            return self._abstain("llm_abstain")

        parsed = self._critique(parsed, evidence)
        if not parsed.get("clinician_bullets"):
            return self._abstain("critique_rejected_all")

        return PipelineResult(
            status="answer",
            clinician_bullets=parsed["clinician_bullets"],
            patient_bullets=parsed.get("patient_bullets", []),
            citations=_collect_citations(chunks),
            confidence=best_score,
            evidence_chunks=_to_evidence_chunks(chunks),
            route={},
        )

    def _drug_qa(self, query: str, target_kbs: List[str]) -> PipelineResult:
        kbs = target_kbs if target_kbs else [config.COLLECTION_DRUGLABELS]
        if config.COLLECTION_DRUGLABELS not in kbs:
            kbs = [config.COLLECTION_DRUGLABELS] + kbs

        chunks = self.retriever.retrieve(query, kbs)
        if not chunks:
            return self._abstain("no_drug_chunks_retrieved")

        best_score = chunks[0]["score"]
        if best_score < config.KB_SIM_THRESHOLD.get(config.COLLECTION_DRUGLABELS, 0.5):
            return self._abstain(f"low_confidence ({best_score:.3f})")

        evidence = _build_evidence_block(chunks)
        raw = llm.complete(_DRUG_SYSTEM, _DRUG_PROMPT.format(query=query, evidence=evidence),
                            config.MAX_TOKENS_ANSWER, config.OPENAI_MODEL)
        parsed = llm.parse_json(raw)

        if not parsed or parsed.get("status") == "abstain" or not parsed.get("clinician_bullets"):
            return self._abstain("llm_abstain")

        return PipelineResult(
            status="answer",
            clinician_bullets=parsed["clinician_bullets"],
            patient_bullets=parsed.get("patient_bullets", []),
            citations=_collect_citations(chunks),
            confidence=best_score,
            evidence_chunks=_to_evidence_chunks(chunks),
            route={},
        )

    def _summarize(self, target_kbs: List[str]) -> PipelineResult:
        store_name = config.COLLECTION_USER if config.COLLECTION_USER in target_kbs else (
            target_kbs[0] if target_kbs else None)
        if not store_name:
            return self._abstain("no_target_kb_for_summarize")

        chunks = self.retriever.stratified_sample(store_name, n=16)
        if not chunks:
            return self._abstain("empty_store_for_summarize")

        evidence = _build_evidence_block(chunks)
        raw = llm.complete(_SUMMARIZE_SYSTEM, _SUMMARIZE_PROMPT.format(evidence=evidence),
                            config.MAX_TOKENS_SUMMARIZE, config.OPENAI_MODEL)
        parsed = llm.parse_json(raw)

        if not parsed or not parsed.get("clinician_bullets"):
            return self._abstain("summarize_llm_empty")

        return PipelineResult(
            status="answer",
            clinician_bullets=parsed["clinician_bullets"],
            patient_bullets=parsed.get("patient_bullets", []),
            citations=_collect_citations(chunks),
            confidence=1.0,
            evidence_chunks=_to_evidence_chunks(chunks),
            route={},
        )

    def _critique(self, parsed: Dict, evidence: str) -> Dict:
        draft = {
            "clinician_bullets": parsed.get("clinician_bullets", []),
            "patient_bullets": parsed.get("patient_bullets", []),
        }
        import json as _json
        raw = llm.complete(_CRITIQUE_SYSTEM,
                            _CRITIQUE_PROMPT.format(draft=_json.dumps(draft, indent=2), evidence=evidence),
                            config.MAX_TOKENS_ANSWER, config.OPENAI_MODEL)
        result = llm.parse_json(raw)
        return result if result else parsed
