"""
aligned_backend.py

Backend engine for Streamlit UI:
- Retrieval + gates via rag_pipeline_aligned.py
- Product-friendly summaries (paragraph) without adding new facts
- Paper-style traceability: Claim -> Supporting Snippet -> Citation
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import time
import re

from config import (
    FAISS_INDICES_DIR,
    KB_GUIDELINES,
    KB_DRUGLABELS,
    KB_USER_FACT,
    TOP_K,
    ZERO_HALLUCINATION_MODE,
)

from storage_utils import create_faiss_store
from embedding_utils import MedCPTDualEmbedder
from rag_pipeline_aligned import RAGPipeline

import rag_pipeline_aligned as rpa
from router import RouteDecision

from pdf_utils import extract_text_by_page
from chunking_utils import chunk_document


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _ntotal(store) -> int:
    try:
        return int(store.index.ntotal)
    except Exception:
        return -1


def bullets_to_paragraph(text: str, max_sentences: int = 7) -> str:
    t = (text or "").replace("FINAL:", "").strip()
    if not t:
        return ""

    sentences: List[str] = []
    for ln in t.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if ln.startswith("- "):
            ln = ln[2:].strip()
        ln = re.sub(r"^[-•\*]\s+", "", ln).strip()
        if not ln:
            continue
        if len(ln) > 10 and not ln.endswith((".", ";")):
            ln += "."
        sentences.append(ln)

    if not sentences:
        return ""

    sentences = sentences[:max_sentences]
    if len(sentences) <= 4:
        return " ".join(sentences).replace(" .", ". ").replace("  ", " ").strip()

    p1 = " ".join(sentences[:3]).strip()
    p2 = " ".join(sentences[3:]).strip()
    return (p1 + "\n\n" + p2).strip()


def _extract_claim_bullets(clinician_bullets: str) -> List[str]:
    t = (clinician_bullets or "").replace("FINAL:", "").strip()
    claims = []
    for ln in t.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if ln.startswith("- "):
            ln = ln[2:].strip()
        ln = re.sub(r"^[-•\*]\s+", "", ln).strip()
        if ln:
            claims.append(ln)
    return claims


def _split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    # conservative split
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = []
    for p in parts:
        p = p.strip()
        if len(p) >= 20:
            out.append(p)
    return out[:40]  # cap


_STOP = {"the","a","an","of","to","in","for","with","on","at","by","is","are","was","were","be","has","have","had","and","or","as","it","this","that","these","those","from"}


def _token_set(s: str) -> set:
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-/%]*", (s or "").lower())
    return {t for t in toks if t not in _STOP and len(t) > 2}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def align_claims_to_snippets(
    claims: List[str],
    evidence_chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Deterministic alignment:
    - For each claim, find best sentence from evidence chunks by token Jaccard overlap.
    - Returns list of {claim, snippet, citation, chunk_store, score}
    """
    aligned = []
    for claim in claims:
        claim_tokens = _token_set(claim)

        best = {
            "score": 0.0,
            "snippet": "",
            "citation": "",
            "store": "",
        }

        for ch in evidence_chunks or []:
            ch_text = (ch.get("text") or "")
            citation = ch.get("citation") or "Unknown source"
            store = ch.get("store") or ""

            for sent in _split_sentences(ch_text):
                score = _jaccard(claim_tokens, _token_set(sent))
                if score > best["score"]:
                    best = {
                        "score": float(score),
                        "snippet": sent.strip(),
                        "citation": citation,
                        "store": store,
                    }

        # If nothing matched well, fall back to the first chunk's first sentence (still transparent)
        if not best["snippet"] and evidence_chunks:
            fallback_text = evidence_chunks[0].get("text", "")
            sents = _split_sentences(fallback_text)
            if sents:
                best["snippet"] = sents[0]
                best["citation"] = evidence_chunks[0].get("citation", "Unknown source")
                best["store"] = evidence_chunks[0].get("store", "")
                best["score"] = 0.0

        aligned.append(
            {
                "claim": claim,
                "snippet": best["snippet"],
                "citation": best["citation"],
                "store": best["store"],
                "score": best["score"],
            }
        )

    return aligned


# -----------------------------
# Ingestion for uploaded PDFs
# -----------------------------
def ingest_user_file(pdf_path: Path) -> None:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"User upload not found: {pdf_path}")

    pages = extract_text_by_page(str(pdf_path))
    if not pages:
        return

    all_chunks: List[Dict[str, Any]] = []
    for page_number, page_text in pages:
        if not page_text or not page_text.strip():
            continue

        chunks = chunk_document(
            text=page_text,
            document_name=pdf_path.name,
            page_number=page_number,
            chunk_size=400,
            overlap=50,
        )

        for c in chunks:
            meta = c.get("metadata", {}) or {}
            meta.update(
                {
                    "source_type": "user_upload",
                    "organization": "User Upload",
                    "source": "User Upload",
                    "document": pdf_path.name,
                    "file_name": pdf_path.name,
                    "section": "Uploaded Document",
                    "section_title": "Uploaded Document",
                    "page_numbers": [page_number] if page_number is not None else [],
                    "file_path": str(pdf_path),
                    "ingested_at": _now_str(),
                }
            )
            c["metadata"] = meta

        all_chunks.extend(chunks)

    if not all_chunks:
        return

    texts = [c["text"] for c in all_chunks]
    metas = [c["metadata"] for c in all_chunks]

    embedder = MedCPTDualEmbedder()
    vectors = embedder.embed_texts(texts)

    store = create_faiss_store(
        store_name=KB_USER_FACT,
        dimension=vectors.shape[1],
        base_dir=str(FAISS_INDICES_DIR),
        embedder=embedder,
    )
    store.add_vectors(vectors, metas)
    store.save()


# -----------------------------
# Streamlit-facing Engine
# -----------------------------
class AlignedScholarBotEngine:
    def __init__(self, verbose: bool = False, print_kb_stats: bool = True):
        self.verbose = verbose
        self.embedder = MedCPTDualEmbedder()

        self.guidelines_store = create_faiss_store(
            store_name=KB_GUIDELINES, dimension=1024, base_dir=str(FAISS_INDICES_DIR), embedder=self.embedder
        )
        self.druglabels_store = create_faiss_store(
            store_name=KB_DRUGLABELS, dimension=1024, base_dir=str(FAISS_INDICES_DIR), embedder=self.embedder
        )
        self.user_store = create_faiss_store(
            store_name=KB_USER_FACT, dimension=1024, base_dir=str(FAISS_INDICES_DIR), embedder=self.embedder
        )

        self.kb_guidelines = self.guidelines_store
        self.kb_druglabels = self.druglabels_store
        self.user_kb = self.user_store

        if print_kb_stats:
            print(f"[KB] FAISS_INDICES_DIR = {Path(FAISS_INDICES_DIR).resolve()}")
            g_n = _ntotal(self.kb_guidelines)
            d_n = _ntotal(self.kb_druglabels)
            u_n = _ntotal(self.user_kb)
            print(f"[KB] guidelines ntotal = {g_n}")
            print(f"[KB] druglabels ntotal = {d_n}")
            print(f"[KB] user_fact ntotal = {u_n}")

        self.pipeline = RAGPipeline(
            query_embedder=self.embedder,
            kb_guidelines_store=self.kb_guidelines,
            kb_druglabels_store=self.kb_druglabels,
            user_kb_store=self.user_kb,
            top_k=TOP_K,
            verbose=self.verbose,
            logger=None,
        )

    def reload_user_kb(self) -> None:
        """Dynamically refresh the user FAISS index from disk without rebooting the server cache."""
        self.user_store = create_faiss_store(
            store_name=KB_USER_FACT, dimension=768, base_dir=str(FAISS_INDICES_DIR), embedder=self.embedder
        )
        self.user_kb = self.user_store
        self.pipeline.user_kb = self.user_store
        if self.verbose:
            print(f"[KB] Reloaded user_fact into memory (ntotal = {_ntotal(self.user_kb)})")

    def generate_response(
        self,
        query: str,
        model_name: str = "llama3",
        force_user_kb: bool = False,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, float, Dict[str, Any]]:
        history = history or []
        _ = history
        _ = model_name

        g_n = _ntotal(self.kb_guidelines)
        d_n = _ntotal(self.kb_druglabels)
        if self.kb_guidelines is None or self.kb_druglabels is None or g_n <= 0 or d_n <= 0:
            meta = {
                "title": "Evidence-backed answer (Aligned RAG)",
                "status": "error_kb_not_loaded",
                "source": "Unknown",
                "references": [],
                "route": {},
                "zero_hallucination_mode": bool(ZERO_HALLUCINATION_MODE),
                "reason": (
                    "Static KB empty/not loaded. "
                    f"FAISS_INDICES_DIR={Path(FAISS_INDICES_DIR).resolve()} "
                    f"guidelines_ntotal={g_n} druglabels_ntotal={d_n}"
                ),
            }
            return "**Final Answer:** System error: KB not loaded (empty FAISS index).", 0.0, meta

        orig_route_query = rpa.route_query
        try:
            if force_user_kb:
                def _route_user_only(q: str, user_uploaded_available: bool = True) -> RouteDecision:
                    return RouteDecision(
                        intent="user_only",
                        target_kbs=[KB_USER_FACT],
                        preferred_section_groups=[],
                        reason="Force user KB only (Streamlit upload context-lock).",
                        task_hints=[],
                    )
                rpa.route_query = _route_user_only

            result = self.pipeline.retrieve_and_answer(query)
        finally:
            rpa.route_query = orig_route_query

        clinician_bullets = getattr(result, "clinician_answer", "") or getattr(result, "answer", "") or ""
        patient_bullets = getattr(result, "patient_answer", "") or ""
        citations = getattr(result, "citations", []) or []
        confidence = float(getattr(result, "confidence", 0.0) or 0.0)
        status = getattr(result, "status", "unknown")
        route = getattr(result, "route", {}) or {}
        source_kbs = getattr(result, "source_kbs", []) or []
        source_kbs = list(dict.fromkeys(source_kbs))

        clinician_paragraph = bullets_to_paragraph(clinician_bullets)
        patient_paragraph = bullets_to_paragraph(patient_bullets)

        # Evidence chunks for snippet alignment (from pipeline patch)
        debug_info = getattr(result, "debug_info", {}) or {}
        evidence_chunks = debug_info.get("evidence_chunks", []) or []

        # ENHANCEMENT: Populate source metadata for UI
        for chunki in evidence_chunks:
            if not chunki.get("source") or chunki.get("source") == "Unknown":
                raw_meta = chunki.get("metadata", {})
                candidates = [
                    chunki.get("citation"),  # Prioritize formatted citation
                    raw_meta.get("source"),
                    raw_meta.get("title"),
                    raw_meta.get("filename"),
                    raw_meta.get("file_path"),
                    raw_meta.get("document_name"),
                    raw_meta.get("document"),
                ]
                final_src = next((c for c in candidates if c and "Unknown source" not in str(c)), "Medical Guideline/Label")
                chunki["source"] = str(final_src)

        # Claim -> snippet alignment (paper-friendly)
        claims = _extract_claim_bullets(clinician_bullets) if status == "answer" else []
        claim_snippets = align_claims_to_snippets(claims, evidence_chunks) if claims and evidence_chunks else []

        # Main product-friendly response (paragraphs)
        parts: List[str] = []
        
        if clinician_bullets == "OPENAI_API_KEY_MISSING":
            parts.append("**SYSTEM ERROR: OpenAI API Key Not Found or Invalid.** Please provide a valid OPENAI_API_KEY in your `.env` or set it locally before launching the application.")
            status = "error"
        elif status == "abstain" or clinician_bullets.strip().upper() == "ABSTAIN":
            parts.append("**Final Answer:** No confidence in answering based on the available evidence.")
        else:
            parts.append("### Clinician Summary\n" + (clinician_paragraph or clinician_bullets.replace("FINAL:", "").strip()))
            if patient_bullets and patient_bullets.strip().upper() != "ABSTAIN":
                parts.append("\n### Patient Summary\n" + (patient_paragraph or patient_bullets.strip()))

        if citations:
            ref_list = []
            for i, c in enumerate(citations):
                # Professional numbered list for poster presentation
                ref_list.append(f"{i+1}. {c}")
            parts.append("\n### Reference\n" + "\n".join(ref_list))

        response_text = "\n\n".join(parts).strip()

        meta = {
            "title": "Evidence-backed answer (Aligned RAG)",
            "source": " + ".join(source_kbs) if source_kbs else "Unknown",
            "references": citations,
            "status": status,
            "route": route,
            "zero_hallucination_mode": bool(ZERO_HALLUCINATION_MODE),

            # For UI expanders
            "clinician_bullets": clinician_bullets.strip(),
            "patient_bullets": patient_bullets.strip(),
            "clinician_paragraph": clinician_paragraph.strip(),
            "patient_paragraph": patient_paragraph.strip(),

            # NEW: snippet alignment payload
            "claim_snippets": claim_snippets,
            "evidence_chunks": evidence_chunks,
        }

        if force_user_kb and meta.get("status") == "abstain":
            meta["source"] = KB_USER_FACT

        return response_text, confidence, meta
