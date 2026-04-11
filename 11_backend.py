"""
11_backend.py
Engine layer — connects the Streamlit UI to the pipeline.
Architecture position: 11 — the single object the UI instantiates.

Responsibilities:
- Load and hold FAISS stores + retriever
- Own one ContextManager instance per session
- Orchestrate: context resolve → route → pipeline.run → format response
- Expose reload_user_kb() for hot-reloading after upload
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_cfg  = importlib.import_module("01_config")
_emb  = importlib.import_module("02_embedding_utils")
_stor = importlib.import_module("03_storage_utils")
_ctx  = importlib.import_module("07_context_manager")
_rtr  = importlib.import_module("08_router")
_retr = importlib.import_module("09_retriever")
_pipe = importlib.import_module("10_pipeline")


# ── Response formatting ───────────────────────────────────────────────────────

def _bullets_to_paragraph(bullets: List[str], max_sentences: int = 8) -> str:
    """Convert a list of bullet strings to flowing paragraph text."""
    sentences = []
    for b in bullets:
        b = b.strip().lstrip("- •*").strip()
        if b and len(b) > 10:
            if not b.endswith((".", ";")):
                b += "."
            sentences.append(b)

    sentences = sentences[:max_sentences]
    if not sentences:
        return ""
    if len(sentences) <= 3:
        return " ".join(sentences)
    # Two-paragraph split for readability
    return " ".join(sentences[:3]) + "\n\n" + " ".join(sentences[3:])


def _format_response(result) -> str:
    """
    Build the final markdown string shown in the UI from a PipelineResult.
    Structure: Clinician Summary → Patient Summary → References
    """
    if result.status == "abstain":
        reason_map = {
            "out_of_scope":                 "This question is outside ScholarBOT's knowledge domain (TB and Pneumonia).",
            "no_chunks_retrieved":          "No relevant evidence was found in the knowledge base.",
            "low_confidence":               "Retrieved evidence did not meet the confidence threshold.",
            "evidence_insufficient_for_query": "The retrieved evidence does not contain enough information to answer this question.",
            "llm_abstain":                  "The generation model could not produce a grounded answer from the evidence.",
            "critique_rejected_all":        "All generated claims were rejected by the peer-review step as unsupported.",
            "no_drug_chunks_retrieved":     "No relevant drug label evidence was found.",
            "empty_store_for_summarize":    "The document appears to be empty or could not be read.",
            "no_target_kb_for_summarize":   "Please upload a document before requesting a summary.",
            "summarize_llm_empty":          "Could not generate a summary from the document.",
        }
        reason = result.abstain_reason or "insufficient_evidence"
        detail = reason_map.get(reason, "Insufficient evidence to answer with confidence.")
        return f"**No Confidence — Abstaining**\n\n{detail}"

    parts = []

    clin_para = _bullets_to_paragraph(result.clinician_bullets)
    if clin_para:
        parts.append(f"### Clinician Summary\n{clin_para}")

    pat_para = _bullets_to_paragraph(result.patient_bullets)
    if pat_para:
        parts.append(f"### Patient Summary\n{pat_para}")

    if result.citations:
        ref_lines = "\n".join(f"{i+1}. {c}" for i, c in enumerate(result.citations))
        parts.append(f"### References\n{ref_lines}")

    return "\n\n".join(parts).strip()


# ── ScholarBotEngine ──────────────────────────────────────────────────────────

class ScholarBotEngine:
    """
    Main engine. One instance per Streamlit session (cached with st.cache_resource).
    """

    def __init__(self, api_key: str = None, verbose: bool = False):
        self.verbose = verbose

        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        # Shared embedder (singleton)
        self.embedder = _emb.get_embedder()

        # Load FAISS stores
        dim = self.embedder.dim
        self.guidelines_store  = _stor.get_or_create_faiss(_cfg.KB_GUIDELINES,  dim, self.embedder)
        self.druglabels_store  = _stor.get_or_create_faiss(_cfg.KB_DRUGLABELS,  dim, self.embedder)
        self.user_store        = _stor.get_or_create_faiss(_cfg.KB_USER,         dim, self.embedder)

        if verbose:
            print(f"[Engine] guidelines  ntotal = {self._ntotal(self.guidelines_store)}")
            print(f"[Engine] druglabels  ntotal = {self._ntotal(self.druglabels_store)}")
            print(f"[Engine] user_kb     ntotal = {self._ntotal(self.user_store)}")

        # Retriever
        self.retriever = _retr.HybridRetriever(
            stores={
                _cfg.KB_GUIDELINES: self.guidelines_store,
                _cfg.KB_DRUGLABELS: self.druglabels_store,
                _cfg.KB_USER:       self.user_store,
            },
            embedder=self.embedder,
        )

        # Pipeline
        self.pipeline = _pipe.RAGPipeline(self.retriever)

        # Context manager (one per session — tracks conversation topic)
        self.context_manager = _ctx.ContextManager()

    # ── User KB hot-reload ────────────────────────────────────────────────────

    def reload_user_kb(self) -> None:
        """Call after a new PDF is ingested so the retriever sees the fresh index."""
        self.user_store = _stor.get_or_create_faiss(_cfg.KB_USER, self.embedder.dim, self.embedder)
        self.retriever.reload_store(_cfg.KB_USER, self.user_store)
        if self.verbose:
            print(f"[Engine] user_kb reloaded (ntotal = {self._ntotal(self.user_store)})")

    def reset_context(self) -> None:
        """Call when the user clears chat history."""
        self.context_manager.reset()

    # ── Main query entry point ────────────────────────────────────────────────

    def generate_response(
        self,
        query: str,
        force_user_kb: bool = False,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Args:
            query        : Raw user query string.
            force_user_kb: True when UI is in "User Document Only" mode.
            history      : Full st.session_state.messages (excluding current turn).

        Returns:
            (response_text, confidence, meta_dict)
        """
        history = history or []

        # Validate KBs are loaded
        g_n = self._ntotal(self.guidelines_store)
        d_n = self._ntotal(self.druglabels_store)
        if g_n <= 0 or d_n <= 0:
            return (
                "**System Error:** Knowledge bases not loaded. "
                f"Check that faiss_indices/ contains guidelines_kb and druglabels_kb. "
                f"(guidelines={g_n}, druglabels={d_n})",
                0.0,
                {"status": "error"},
            )

        # 1. Context resolution (strip meta-refs, resolve pronouns)
        has_user_doc = self._ntotal(self.user_store) > 0
        resolved = self.context_manager.resolve(query, history)
        if resolved != query and self.verbose:
            print(f"[Engine] Query resolved: '{query}' -> '{resolved}'")

        # 2. Route
        route = _rtr.route(resolved, has_user_doc=has_user_doc, force_user_kb=force_user_kb)

        # 3. Run pipeline
        result = self.pipeline.run(resolved, route)
        result.route = route

        # 4. Update context topic for future turns
        self.context_manager.update_topic(resolved, result.status)

        # 5. Format response
        response_text = _format_response(result)
        confidence    = result.confidence

        meta = {
            "status":          result.status,
            "abstain_reason":  result.abstain_reason,
            "source":          " + ".join(route.get("target_kbs", [])),
            "references":      result.citations,
            "route":           route,
            "evidence_chunks": result.evidence_chunks,
            "clinician_bullets": result.clinician_bullets,
            "patient_bullets":   result.patient_bullets,
        }

        return response_text, confidence, meta

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _ntotal(store) -> int:
        try:
            return int(store.index.ntotal)
        except Exception:
            return -1
