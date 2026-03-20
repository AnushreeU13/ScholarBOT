"""
rag_pipeline_aligned.py

Tier-1 RAG pipeline (Hybrid + Gates, stable v4)

Major fixes:
A) Router task_hints separation (no more preferred pollution like "diagnosis" inside section groups)
B) Guideline Diagnosis Gate (hard filter for diagnosis/testing evidence when asked)
C) Drug Anchor Filter (unlocked) to prevent wrong-drug retrieval drift
D) Entailment Gate for clinician bullets (lightweight lexical entailment + high-risk term blocking)
E) Patient Safety Gate (medical-entity-based) + stronger patient prompt to avoid "clinician copy / background explanations"

Dependencies:
- config.py / config_local_v2.py
- router.py
- llm_utils.py
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Set
import re
import time
import numpy as np
import torch

# ------------------------------------------------------------
# Config Import
# ------------------------------------------------------------
from config import (
    KB_SIM_THRESHOLD, DEFAULT_SIM_THRESHOLD, TOP_K,
    ZERO_HALLUCINATION_MODE, USE_CLINICIAN_LLM,
    KB_DRUGLABELS, KB_GUIDELINES, KB_USER_FACT,
    USE_QUERY_EXPANSION, USE_RERANKER, RERANK_MODEL_NAME, RERANK_K,
    USE_HYBRID_SEARCH, STRICT_USER_CONTEXT, USE_SELF_CRITIQUE,
    TOP_K_DENSE, TOP_K_SPARSE
)

# ------------------------------------------------------------
# Router Import / Mock
# ------------------------------------------------------------
try:
    from router import route_query, RouteDecision
except Exception:
    @dataclass
    class RouteDecision:
        intent: str
        target_kbs: List[str]
        preferred_section_groups: List[str]
        reason: str
        task_hints: List[str] = None

    def route_query(query: str, user_uploaded_available: bool = False) -> RouteDecision:
        targets = ["kb_druglabels_medcpt", "kb_guidelines_medcpt"]
        if user_uploaded_available:
            targets.append("user_fact_kb_medcpt")
        return RouteDecision("mixed", targets, [], "Fallback router.", task_hints=[])

# ------------------------------------------------------------
# Data Structures
# ------------------------------------------------------------
@dataclass
class RAGResult:
    answer: str
    clinician_answer: str
    patient_answer: str
    citations: List[str]
    confidence: float
    status: str
    source_kbs: List[str]
    route: Dict[str, Any]
    debug_info: Dict[str, Any]
    consistency: Dict[str, Any]

# ------------------------------------------------------------
# LLM Helpers
# ------------------------------------------------------------
def _clean_generated_text(raw_text: str, marker: str = "OUTPUT:") -> str:
    if marker in raw_text:
        raw_text = raw_text.split(marker, 1)[-1]
    raw_text = re.sub(r"(?im)^\s*(system|user|assistant)\s*:?\s*", "", raw_text)
    raw_text = re.sub(r"[\u4e00-\u9fff]", "", raw_text)  # strip Chinese artifacts
    return raw_text.strip()

def _generate_with_prompt(prompt: str, max_new_tokens: int = 256) -> str:
    """
    Hybrid generator:
    1) If OPENAI_API_KEY exists and OpenAI SDK is installed -> use OpenAI (default: gpt-4o-mini)
    2) Otherwise -> fallback to local transformers model (llm_utils: tokenizer/model/clean_llm_answer)

    Returns raw text (caller will clean/parse).
    Fail-closed behavior: if BOTH fail, returns a sentinel string "GENERATOR_ERROR: ..."
    """
    import os
    import re

    try:
        from config import LOCAL_QA_PROMPT as system_msg
    except ImportError:
        system_msg = "You are a clinical assistant. Answer in complete English sentences. Use ONLY provided evidence."

    # -------------------------
    # 1) Try OpenAI (preferred)
    # -------------------------
    use_openai = bool(os.getenv("OPENAI_API_KEY"))
    if use_openai:
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI()
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=max_new_tokens,
            )

            if resp and resp.choices:
                return resp.choices[0].message.content.strip()

            # If OpenAI returned empty, fall back to local
        except Exception as e:
            # OpenAI failed -> fall back to local
            pass

    # -------------------------
    # 2) Local fallback (Try Ollama First)
    # -------------------------
    try:
        from langchain_community.chat_models import ChatOllama
        from langchain_core.messages import SystemMessage, HumanMessage
        
        # User requested "llama3" 
        ollama = ChatOllama(model="llama3", temperature=0)
        
        msgs = [
            SystemMessage(content=system_msg),
            HumanMessage(content=prompt)
        ]
        resp = ollama.invoke(msgs)
        
        if resp and resp.content:
            return resp.content.strip()
    except Exception as e:
        # If Ollama fails (not installed/running), proceed to local Qwen
        pass

    # -------------------------
    # 3) Local fallback (Qwen transformers)
    # -------------------------
    try:
        from llm_utils import tokenizer, model, clean_llm_answer  # your local loader

        # If chat template exists, format like chat
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"{system_msg}\n\n{prompt}"

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs.input_ids.shape[1]

        import torch
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = out[0][input_len:]
        raw = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        raw = clean_llm_answer(raw)

        # Basic cleanup (avoid odd role artifacts)
        raw = re.sub(r"(?im)^\s*(system|user|assistant)\s*:?\s*", "", raw).strip()
        return raw if raw else "GENERATOR_ERROR: local_empty"

    except Exception as e:
        return f"GENERATOR_ERROR: {type(e).__name__}"


# ------------------------------------------------------------
# v7 Features: Query Expansion & Reranking
# ------------------------------------------------------------

def _expand_query(query: str) -> List[str]:
    """
    Generates 2 clinical variations of the query to improve recall.
    """
    prompt = f"""
    Task: Generate 2 short clinical variations of the user's question to help find more relevant medical documents.
    User Question: {query}
    
    Rules:
    1) Provide only the variations, one per line.
    2) Keep them clinically relevant and precise.
    3) Do not add explanations.
    
    VARIATIONS:
    """
    raw = _generate_with_prompt(prompt, max_new_tokens=100)
    lines = [ln.strip("- ").strip() for ln in raw.splitlines() if len(ln.strip()) > 5]
    return [query] + lines[:2]

_RERANKER_MODEL = None

def _rerank_candidates(query: str, candidates: List[Dict], k: int = 8) -> List[Dict]:
    """
    Uses a Cross-Encoder to re-score the top candidates.
    """
    if not candidates: return []
    
    global _RERANKER_MODEL
    try:
        from sentence_transformers import CrossEncoder
        if _RERANKER_MODEL is None:
            print(f"[RAG] Loading Reranker: {RERANK_MODEL_NAME}...")
            _RERANKER_MODEL = CrossEncoder(RERANK_MODEL_NAME)
            
        # Prepare pairs
        pairs = [[query, c["text"]] for c in candidates]
        scores = _RERANKER_MODEL.predict(pairs)
        
        # Update scores
        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score)
            
        # Re-sort
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:k]
    except Exception as e:
        print(f"[RAG] Reranking failed: {e}. Falling back to similarity scores.")
        return candidates[:k]



# ------------------------------------------------------------
# Constants (anchors / domain)
# ------------------------------------------------------------
_DRUG_ANCHORS = {
    "isoniazid","rifampin","rifampicin","pyrazinamide","ethambutol",
    "azithromycin","levofloxacin","moxifloxacin","amoxicillin","linezolid"
}
_DOMAIN_TERMS = {"tuberculosis","tb","pneumonia","cap","community","acquired","ats","idsa","who","cdc"}

# ------------------------------------------------------------
# Utility: cleaning & citations
# ------------------------------------------------------------
def _clean_pdf_text(s: str) -> str:
    if not isinstance(s, str): return ""
    t = s.replace("\u00ad", "")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _stable_citation(meta: Dict[str, Any]) -> str:
    if not meta: return "Unknown source"
    if meta.get("source") == "DailyMed" or meta.get("doc_type") == "druglabel_spl":
        title = meta.get("title") or meta.get("source_title") or "DrugLabel"
        section = meta.get("section_title") or meta.get("section_group") or "Section"
        date = meta.get("source_date") or meta.get("upload_date") or "n/a"
        return f"DailyMed | {title} | {section} | {date}"
    doc = meta.get("document") or meta.get("title") or "Guideline"
    sec = meta.get("section") or meta.get("section_title") or ""
    year = meta.get("year") or "n/a"
    return f"{doc} | {sec} | {year}"

def _collect_citations(chunks: List[Dict], max_items: int = 5) -> List[str]:
    seen = set()
    out = []
    for c in chunks:
        cit = _stable_citation(c.get("metadata", {}))
        if cit not in seen:
            seen.add(cit)
            out.append(cit)
    return out[:max_items]

def _section_group_from_meta(meta: Dict) -> str:
    sec = (meta.get("section") or meta.get("section_title") or "").lower()
    grp = (meta.get("section_group") or "").lower()
    if grp:
        return grp

    # --- Drug label style groups (existing) ---
    if any(k in sec for k in ["dose", "dosage", "admin", "administration"]): 
        return "dosage"
    if any(k in sec for k in ["contraindication"]): 
        return "contraindications"
    if any(k in sec for k in ["warn", "precaution", "boxed", "black box"]): 
        return "warnings"
    if any(k in sec for k in ["adverse", "side effect", "reaction", "toxicity"]): 
        return "adverse"
    if "interact" in sec or "cyp" in sec: 
        return "interactions"
    if any(k in sec for k in ["indication", "indications", "use in"]): 
        return "indications"

    # --- Guideline-oriented groups (NEW) ---
    # Treatment / regimen / recommendation sections
    if any(k in sec for k in [
        "recommendation", "recommendations",
        "treatment", "therapy", "management",
        "regimen", "regimens", "duration", "course",
        "follow-up", "monitoring", "rationale"
    ]):
        return "g_treatment"

    # Diagnosis / testing / evaluation sections
    if any(k in sec for k in [
        "diagnos", "diagnosis", "testing", "test", "evaluation", "work-up",
        "radiograph", "x-ray", "imaging", "culture", "sputum", "pcr"
    ]):
        return "g_diagnosis"

    # Prevention / infection control
    if any(k in sec for k in [
        "prevention", "prevent", "prophylaxis",
        "infection control", "vaccin"
    ]):
        return "g_prevention"

    return "other"


def _apply_section_bias(sims: np.ndarray, metas: List[Dict], preferred_groups: List[str], texts: List[str] = None, boost: float = 0.12) -> np.ndarray:
    if len(sims) == 0: return sims
    boosted = sims.astype(np.float32).copy()
    pref_set = {p.lower() for p in preferred_groups}

    for i, meta in enumerate(metas):
        add = 0.0
        if _section_group_from_meta(meta) in pref_set:
            add += boost

        if texts:
            txt = (texts[i] or "").lower()[:800]
            # downrank obvious references
            if "references" in txt[:60] or "bibliography" in txt[:60] or "doi:" in txt:
                add -= 0.5

        boosted[i] += add

    return np.clip(boosted, 0.0, 1.0)

# ------------------------------------------------------------
# Gate C: Drug Anchor Filtering (Unlocked)
# ------------------------------------------------------------
def _filter_candidates_by_drug_anchor(candidates: List[Dict], query: str) -> List[Dict]:
    q = (query or "").lower()
    found = [d for d in _DRUG_ANCHORS if d in q]
    if not found:
        return candidates

    filtered = []
    for c in candidates:
        if c["store"] != KB_DRUGLABELS:
            filtered.append(c)
            continue
        title = (c["metadata"].get("title") or c["metadata"].get("source_title") or "").lower()
        text_full = (c["text"] or "").lower()  # unlocked: no truncation
        if any(d in title for d in found) or any(d in text_full for d in found):
            filtered.append(c)
    return filtered

# ------------------------------------------------------------
# Gate B: Guideline Diagnosis Gate
# ------------------------------------------------------------
_DIAG_KEYS = ["diagnos", "testing", "test", "workup", "radiograph", "x-ray", "imaging", "culture", "sputum", "blood", "procalcitonin", "pcr"]

def _guideline_diagnosis_gate(chunks: List[Dict]) -> List[Dict]:
    out = []
    for c in chunks:
        meta = c.get("metadata", {})
        sec = (meta.get("section_title") or meta.get("section") or "").lower()
        txt = (c.get("text") or "").lower()[:900]
        if any(k in sec for k in _DIAG_KEYS) or any(k in txt for k in _DIAG_KEYS):
            out.append(c)
    return out

# ------------------------------------------------------------
# Regex extractors
# ------------------------------------------------------------
def _extract_regex_items(text: str, mode: str) -> List[str]:
    text = _clean_pdf_text(text)
    patterns = []
    if mode == "adr":
        patterns = [
            r"include[^:]{0,80}:\s*(.+?)(?:\.|$)",
            r"following:\s*(.+?)(?:\.|$)",
        ]
    elif mode == "dosage":
        patterns = [r"(\b\d+(?:\.\d+)?\s*mg(?:/kg)?\b.*?(?:\.|$))"]

    items: List[str] = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            content = (m.group(1) or "").strip()
            parts = re.split(r",|;|\band\b", content)
            items.extend([p.strip() for p in parts if len(p.strip()) > 3])

    # de-dup
    seen = set()
    out = []
    for it in items:
        key = re.sub(r"\W+", "", it.lower())
        if key and key not in seen:
            seen.add(key)
            out.append(it)
    return out[:12]

def _parse_relaxed_bullets(text: str, max_items: int = 12) -> List[str]:
    """Smart bullet parser that handles multi-line answers and various marker types."""
    if not isinstance(text, str): return []
    
    # 1. Identify where bullets start
    lines = text.splitlines()
    raw_items = []
    current_item = []
    
    bullet_marker_pattern = re.compile(r"^([-•\*]|\(?\d+\)?[.)])\s+(.*)")
    
    for ln in lines:
        ln_strip = ln.strip()
        if not ln_strip: continue
        
        match = bullet_marker_pattern.match(ln_strip)
        if match:
            # New bullet starts
            if current_item:
                raw_items.append(" ".join(current_item))
            current_item = [match.group(2).strip()]
        else:
            # Continuation of previous bullet or a line without marker
            if current_item:
                current_item.append(ln_strip)
            else:
                # No bullet yet, but text exists - treat as first item
                current_item = [ln_strip]
                
    if current_item:
        raw_items.append(" ".join(current_item))
    
    # 2. Cleanup and Deduplicate
    seen = set()
    out = []
    for it in raw_items:
        it = re.sub(r"\s+", " ", it).strip()
        if not it or len(it) < 5: continue
        
        key = re.sub(r"\W+", "", it.lower())
        if key and key not in seen:
            seen.add(key)
            out.append(it)
            
    # Fallback to splitting if no bullets found at all
    if not out and len(text) < 700:
        parts = re.split(r",|;|\band\b| or ", text)
        out = [p.strip() for p in parts if len(p.strip()) > 3]
        
    return out[:max_items]

# ------------------------------------------------------------
# Gate D: Entailment Gate (tuned)
# ------------------------------------------------------------
_STOP = {"the","a","an","of","to","in","for","with","on","at","by","is","are","was","were","be","has","have","had","and","or"}
_HIGH_RISK = ["carbapenem", "ciprofloxacin", "protease", "hiv", "hepatitis c", "hepatitis b", "mri"]

def _verify_entailment(bullet: str, evidence_text: str, threshold: float = 0.25) -> bool:
    if not bullet:
        return False
    b = bullet.lower().strip()
    e = (evidence_text or "").lower()

    # hard block high-risk additions
    for w in _HIGH_RISK:
        if w in b and w not in e:
            return False

    words = [w.lower() for w in re.findall(r"\w+", bullet) if w.lower() not in _STOP and len(w) > 2]
    if not words:
        return True

    hits = sum(1 for w in words if w in e)
    overlap = hits / max(1, len(words))
    return overlap >= threshold

# ------------------------------------------------------------
# Gate E: Patient Safety Gate (medical-entity-based)
# ------------------------------------------------------------
def _extract_critical_tokens(text: str) -> Set[str]:
    toks = set()
    for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-/%]*", text or ""):
        lw = w.lower()

        # dosage-like (numbers)
        if any(ch.isdigit() for ch in w):
            toks.add(lw); continue

        # all-caps abbreviation (HIV, CAP)
        if len(w) >= 2 and w.isupper():
            toks.add(lw); continue

        # anchors
        if lw in _DRUG_ANCHORS or lw in _DOMAIN_TERMS:
            toks.add(lw); continue

    return toks

def _check_patient_safety(clinician_text: str, patient_text: str) -> bool:
    c = _extract_critical_tokens(clinician_text)
    p = _extract_critical_tokens(patient_text)

    # allow generic words only if they appear as "critical" (rare)
    whitelist = {
        "patient","doctor","medicine","symptoms","treatment","diagnosis","common","severe","mild",
        "acute","chronic","oral","tablet","capsule","daily","weekly"
    }

    new_tokens = [t for t in p if (t not in c and t not in whitelist)]
    if new_tokens:
        print(f"[RAG] Patient Guard flagged new entities: {new_tokens}")
        return False
    return True

def _patient_rewrite_deterministic(clinician_answer: str) -> str:
    if "ABSTAIN" in clinician_answer:
        return "I could not find an explicit answer in the retrieved documents."
    clean = clinician_answer.replace("FINAL:", "").strip()
    return f"From the documents:\n{clean}"

def _consolidate_context(chunks: List[Dict], max_final: int = 8) -> List[Dict]:
    """
    v8 Feature: Merges adjacent or overlapping chunks from the same document/page.
    """
    if not chunks: return []
    
    # Sort by document and chunk index
    # We use a key that includes document name, page, and chunk index
    def sort_key(c):
        m = c.get("metadata", {})
        return (str(m.get("document_name", "")), int(m.get("page_number", 0) or 0), int(m.get("chunk_index", 0) or 0))
    
    sorted_chunks = sorted(chunks, key=sort_key)
    
    merged = []
    if not sorted_chunks: return []
    
    curr = sorted_chunks[0].copy()
    
    for i in range(1, len(sorted_chunks)):
        next_c = sorted_chunks[i]
        
        c_meta = curr.get("metadata", {})
        n_meta = next_c.get("metadata", {})
        
        # Check if same document and adjacent/same chunk index
        same_doc = c_meta.get("document_name") == n_meta.get("document_name")
        same_page = c_meta.get("page_number") == n_meta.get("page_number")
        # Adjacent if chunk_index is within 1
        is_adjacent = abs(int(c_meta.get("chunk_index", 0) or 0) - int(n_meta.get("chunk_index", 0) or 0)) <= 1
        
        if same_doc and same_page and is_adjacent:
            # Merge text (simple concatenation if chunking overlap is handled, 
            # but here we just append unique parts or just full text for safety)
            # Since we have overlap, we'll just concatenate and the LLM will handle redundancy
            if next_c["text"] not in curr["text"]:
                curr["text"] += "\n" + next_c["text"]
            # Keep highest score
            curr["score"] = max(curr["score"], next_c["score"])
        else:
            merged.append(curr)
            curr = next_c.copy()
            
    merged.append(curr)
    
    # Sort back by score and limit
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:max_final]

# ------------------------------------------------------------
# Main Pipeline
# ------------------------------------------------------------
class RAGPipeline:
    def __init__(self, query_embedder, kb_guidelines_store, kb_druglabels_store, user_kb_store=None, top_k=TOP_K, verbose=True, logger=None):
        self.query_embedder = query_embedder
        self.kb_guidelines = kb_guidelines_store
        self.kb_druglabels = kb_druglabels_store
        self.user_kb = user_kb_store
        self.top_k = top_k
        self.verbose = verbose
        self.logger = logger
        self.zero_hallucination_mode = ZERO_HALLUCINATION_MODE
        self.bm25_indices = {}
        
        # Build initial BM25 indices if enabled
        if USE_HYBRID_SEARCH:
            self._build_bm25_indices()

    def _build_bm25_indices(self):
        """Builds lightweight BM25 retrievers on existing text documents."""
        from rank_bm25 import BM25Okapi
        for store in [self.kb_guidelines, self.kb_druglabels, self.user_kb]:
            if store and hasattr(store, 'docstore') and hasattr(store.docstore, '_dict'):
                name = "guidelines" if store == self.kb_guidelines else ("druglabels" if store == self.kb_druglabels else "user")
                # Retrieve all texts from docstore
                docs = list(store.docstore._dict.values())
                if not docs: continue
                
                texts = [d.page_content for d in docs]
                tokenized = [(re.findall(r'\w+', t.lower())) for t in texts]
                if not tokenized: continue
                
                self._log(f"[v9] Building BM25 index for {name} ({len(texts)} docs)...")
                self.bm25_indices[name] = {
                    "bm25": BM25Okapi(tokenized),
                    "docs": docs
                }

    def _bm25_search(self, query: str, store_name: str, k: int = 10) -> List[Dict]:
        name_map = {KB_DRUGLABELS: "druglabels", KB_GUIDELINES: "guidelines", KB_USER_FACT: "user"}
        idx_key = name_map.get(store_name)
        if not idx_key or idx_key not in self.bm25_indices:
            return []
            
        bm25_obj = self.bm25_indices[idx_key]["bm25"]
        original_docs = self.bm25_indices[idx_key]["docs"]
        
        tokenized_q = re.findall(r'\w+', query.lower())
        scores = bm25_obj.get_scores(tokenized_q)
        top_idx = np.argsort(scores)[::-1][:k]
        
        results = []
        for i in top_idx:
            if scores[i] <= 0: continue
            doc = original_docs[i]
            norm_score = float(scores[i]) / (max(scores) if max(scores) > 0 else 1.0)
            results.append({
                "score": norm_score,
                "raw_sim": norm_score,
                "text": doc.page_content,
                "metadata": doc.metadata,
                "store": store_name,
                "type": "sparse"
            })
        return results

    def _log(self, msg: str):
        if self.verbose:
            (self.logger if self.logger else print)(msg)

    def _get_store_by_name(self, name: str):
        if name == KB_DRUGLABELS or name == "kb_druglabels_medcpt": return self.kb_druglabels
        if name == KB_GUIDELINES or name == "kb_guidelines_medcpt": return self.kb_guidelines
        if name == KB_USER_FACT or name == "user_fact_kb_medcpt": return self.user_kb
        return None

    def _check_consistency(self, clinician_text: str, patient_text: str) -> Dict[str, Any]:
        if "ABSTAIN" in clinician_text or "could not find" in patient_text.lower():
            return {"score": 1.0, "status": "pass (abstain)"}

        vec_c = self.query_embedder.embed_query(clinician_text)
        vec_p = self.query_embedder.embed_query(patient_text)

        # normalize cosine-ish
        nc = np.linalg.norm(vec_c)
        npv = np.linalg.norm(vec_p)
        if nc > 0: vec_c = vec_c / nc
        if npv > 0: vec_p = vec_p / npv

        sim = float(np.dot(vec_c, vec_p.T))
        return {"score": sim, "status": "pass" if sim >= 0.72 else "fail"}

    def retrieve_and_answer(self, query: str) -> RAGResult:
        self._log(f"\n=== TIER-1 RAG QUERY (v7): {query} ===")
        t0 = time.time()

        # v7: Query Expansion
        search_queries = [query]
        if USE_QUERY_EXPANSION:
            search_queries = _expand_query(query)
            self._log(f"[QUERY] Expanded: {search_queries}")

        decision = route_query(query, user_uploaded_available=(self.user_kb is not None))

        # v9: Strict Context Locking
        if STRICT_USER_CONTEXT and self.user_kb:
            # Check if user KB has content
            u_n = -1
            try: u_n = int(self.user_kb.index.ntotal)
            except: pass
            if u_n > 0:
                self._log("[v9] STRICT_USER_CONTEXT active. Locking retrieval to user_kb only.")
                decision.target_kbs = [KB_USER_FACT]
                decision.reason = "Strict Mode: Targeted User Document."

        # task_hints compatible
        task_hints = getattr(decision, "task_hints", None) or []
        self._log(f"[ROUTER] Intent: {decision.intent} | Target: {decision.target_kbs} | Pref: {decision.preferred_section_groups} | Hints: {task_hints}")

        candidates: List[Dict] = []
        for q_sub in search_queries:
            self._log(f"[RAG] Searching for: {q_sub}")
            q_vec = self.query_embedder.embed_query(q_sub)
            
            for kb_name in decision.target_kbs:
                store = self._get_store_by_name(kb_name)
                if not store: continue

                # v9: Hybrid Retrieval (Dense Part)
                try:
                    results = store.similarity_search_with_score_by_vector(q_vec, k=TOP_K_DENSE)
                except Exception:
                    print(f"[ERROR] Engine retrieval failed for {kb_name}")
                    continue

                is_l2 = False
                if hasattr(store, 'index') and hasattr(store.index, 'metric_type'):
                    if store.index.metric_type == 1: is_l2 = True

                for doc, score in results:
                    sim = (1.0 - (score / 2.0)) if is_l2 else score
                    candidates.append({
                        "score": float(sim),
                        "raw_sim": float(sim),
                        "text": doc.page_content,
                        "metadata": doc.metadata,
                        "store": kb_name,
                        "type": "dense"
                    })
                
                # v9: Hybrid Retrieval (Sparse Part)
                if USE_HYBRID_SEARCH:
                    sparse_hits = self._bm25_search(q_sub, kb_name, k=TOP_K_SPARSE)
                    # We slightly weight sparse lower unless it's a very exact match
                    for h in sparse_hits:
                        h["score"] = h["score"] * 0.7  # Initial keyword weight
                        candidates.append(h)

        # v9: Apply Section Bias to ALL candidates (Hybrid)
        if candidates:
            sims = np.array([c["score"] for c in candidates], dtype=np.float32)
            metas = [c["metadata"] for c in candidates]
            texts = [c["text"] for c in candidates]
            boosted = _apply_section_bias(sims, metas, decision.preferred_section_groups, texts=texts)
            for i in range(len(candidates)):
                candidates[i]["score"] = float(boosted[i])

        # v7: Multi-query de-duplication

        # v7: Multi-query de-duplication
        seen_texts = set()
        unique_candidates = []
        for c in candidates:
            # Simple text-based deduplication
            t_key = re.sub(r"\W+", "", c["text"][:100]).lower()
            if t_key not in seen_texts:
                seen_texts.add(t_key)
                unique_candidates.append(c)
        candidates = unique_candidates

        # Drug anchor filter (DISABLED per v2 logic)
        # before = len(candidates)
        # candidates = _filter_candidates_by_drug_anchor(candidates, query)
        # after = len(candidates)
        # if before != after:
        #     self._log(f"[RAG] Anchor filter kept {after} candidates (dropped {before-after}).")
        pass

        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_chunks = candidates[:TOP_K]

        # v7: Reranking
        if USE_RERANKER:
            self._log(f"[RAG] Reranking {len(top_chunks)} candidates...")
            final_chunks = _rerank_candidates(query, top_chunks, k=RERANK_K)
        else:
            final_chunks = top_chunks[:RERANK_K]

        # v8: Context Consolidation
        # Merges adjacent/same-page chunks to provide more continuous evidence
        before_merge = len(final_chunks)
        final_chunks = _consolidate_context(final_chunks, max_final=8)
        after_merge = len(final_chunks)
        if before_merge != after_merge:
            self._log(f"[RAG] Context Consolidation: {before_merge} -> {after_merge} merged blocks.")

        # Section isolation for drugs
        if decision.intent == "drug":
            if "adverse" in decision.preferred_section_groups:
                adverse_only = [c for c in top_chunks if c["metadata"].get("section_group") == "adverse"]
                if adverse_only:
                    final_chunks = adverse_only
                    self._log(f"[RAG] Section Isolation: {len(top_chunks)} -> {len(final_chunks)} (adverse)")
            if "interactions" in decision.preferred_section_groups:
                inter_only = [c for c in top_chunks if c["metadata"].get("section_group") == "interactions"]
                if inter_only:
                    final_chunks = inter_only
                    self._log(f"[RAG] Section Isolation: {len(top_chunks)} -> {len(final_chunks)} (interactions)")

        # Guideline diagnosis gate (DISABLED per v2 logic)
        # if decision.intent in ("guideline", "mixed") and ("diagnosis" in task_hints):
        #     gated = _guideline_diagnosis_gate(final_chunks)
        #     if gated:
        #         self._log(f"[RAG] Guideline Diagnosis Gate: {len(final_chunks)} -> {len(gated)}")
        #         final_chunks = gated[:self.top_k]
        #     else:
        #         self._log("[RAG] Guideline Diagnosis Gate: no matching evidence -> Fallback (soft).")
        pass


        if not final_chunks:
            self._log("[RAG] No chunks retrieved (final_chunks empty).")
        
        best_score = final_chunks[0]["score"] if final_chunks else 0.0    
        required = KB_SIM_THRESHOLD.get(final_chunks[0]["store"], 0.65) if final_chunks else 0.7
        if not final_chunks or best_score < required:
            self._log(f"[RAG] Low confidence ({best_score:.3f} < {required}) -> ABSTAIN.")
            return self._build_abstain_result(query, decision)

        evidence_text = "\n\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(final_chunks)])
        # Clean broken PDF wrap lines globally
        evidence_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', evidence_text)
        evidence_text = re.sub(r'\s+\.', '.', evidence_text)
        
        citations = _collect_citations(final_chunks)
        primary_kb = final_chunks[0]["store"]

        clinician_answer = "ABSTAIN"
        mode = "unknown"

        # --- Drug logic ---
        if decision.intent == "drug" and primary_kb == KB_DRUGLABELS:
            fast_mode = None
            if "adverse" in decision.preferred_section_groups:
                fast_mode = "adr"
            elif "dosage" in decision.preferred_section_groups:
                fast_mode = "dosage"

            if fast_mode:
                items = _extract_regex_items(evidence_text, fast_mode)
                if items:
                    clinician_answer = "FINAL:\n" + "\n".join([f"- {x}" for x in items])
                    mode = "regex_deterministic"
                else:
                    self._log("[RAG] Regex empty -> Strict LLM fallback.")
                    clinician_answer = self._generate_clinician_answer(query, evidence_text, intent="drug")
                    mode = "fallback_llm"
            else:
                clinician_answer = self._generate_clinician_answer(query, evidence_text, intent="drug_general")
                mode = "llm_gen"

        # --- Guideline logic ---
        else:
            if USE_CLINICIAN_LLM:
                clinician_answer = self._generate_clinician_answer(query, evidence_text, intent="guideline")
                mode = "guideline_synthesis"
            else:
                clinician_answer = "FINAL:\n" + evidence_text[:1400] + "..."
                mode = "legacy_extract"

        if "ABSTAIN" in clinician_answer:
            return self._build_abstain_result(query, decision)

        # Patient rewrite
        self._log("[RAG] Generating Patient Answer...")
        patient_answer = self._generate_patient_answer(clinician_answer)

        consistency = self._check_consistency(clinician_answer, patient_answer)
        consistency["mode"] = mode
        self._log(f"[CONSISTENCY] Score: {consistency['score']:.3f} | Status: {consistency['status']}")

        evidence_chunks = []
        for c in final_chunks:
            evidence_chunks.append({
                "text": (c.get("text") or "")[:1600],   # chunk text (truncate for UI)
                "citation": _stable_citation(c.get("metadata", {})),
                "store": c.get("store"),
            })


        return RAGResult(
            answer=clinician_answer,
            clinician_answer=clinician_answer,
            patient_answer=patient_answer,
            citations=citations,
            confidence=best_score,
            status="answer",
            source_kbs=[c["store"] for c in final_chunks],
            route={"intent": decision.intent, "task_hints": task_hints},
            debug_info={
                "top_sim_raw": final_chunks[0]["raw_sim"],
                "elapsed_s": round(time.time() - t0, 3),
                "evidence_chunks": evidence_chunks,   # NEW
            },
            consistency=consistency,
        )

    # ---------------------------------------------------------
    # Generation
    # ---------------------------------------------------------
    def _generate_clinician_answer(self, query: str, context: str, intent: str = "general") -> str:
        # strong negatives to reduce drift
        negative = ""
        if intent.startswith("drug"):
            negative = """
Rules additions (drug):
- ONLY extract items explicitly present in EVIDENCE.
- Do NOT add mechanism, background explanations, or new drug names.
- If question is adverse reactions: list reactions/symptoms only (not risk factors/monitoring).
- If question is interactions: list interacting drugs/classes explicitly mentioned.
""".strip()
        else:
            negative = """
Rules additions (guideline):
- ONLY use EVIDENCE.
- Do NOT add symptoms/tests not mentioned.
- Prefer short complete sentences (not fragments).
""".strip()

        prompt = f"""
Task: Answer the QUESTION using the EVIDENCE provided.

Rules:
1) ANSWER FIRST: Start with a direct answer to the question using bullet points.
2) BOILERPLATE: Minimal boilerplate (e.g., "Based on the evidence...") is allowed if it makes reading easier.
3) BULLET FORMAT: Each bullet must start with "- ".
4) EVIDENCE PRIORITIZATION: Prioritize the provided EVIDENCE to build your answer. You may use standard clinical knowledge to define fundamental concepts if they are missing from the evidence, but NEVER contradict the evidence.
5) COMPLETE SENTENCES: Write in complete, cohesive, and grammatical English sentences.
6) COHESIVE STRUCTURE: Avoid line-breaks or fragments mid-sentence. Each point must be meaningful on its own.

{negative}

QUESTION:
{query}

EVIDENCE:
{context[:2800]}

CLINICIAN OUTPUT:
""".strip()

        self._log(f"[DEBUG] LLM PROMPT EVIDENCE:\n{context[:500]}...") # Log first 500 chars

        raw = _generate_with_prompt(prompt, max_new_tokens=240)

        # NEW: fail fast if generator returned an error sentinel
        if raw.startswith("OPENAI_") or raw in ["OPENAI_API_KEY_MISSING", "OPENAI_EMPTY_OUTPUT"]:
            return "ABSTAIN"

        clean = _clean_generated_text(raw, marker="CLINICIAN OUTPUT:")
        if "ABSTAIN" in clean or len(clean) < 5:
            return "ABSTAIN"

        lines = _parse_relaxed_bullets(clean, max_items=12)

        # --- v9: Self-Critique Loop ---
        if USE_SELF_CRITIQUE and len(lines) > 0:
            self._log(f"[v9] Running Self-Critique on {len(lines)} claims...")
            refined_lines = self._refine_answer(query, lines, context)
            lines = refined_lines if refined_lines else lines

        # Bypass verification if strict mode is disabled (Generative Mode)
        if not self.zero_hallucination_mode:
             verified = lines
        else:
            verified: List[str] = []
            for ln in lines:
                if _verify_entailment(ln, context, threshold=0.30):
                    verified.append(ln)
                else:
                    pass
        
        if not verified:
            return "ABSTAIN"

        # Make bullets more sentence-like for clinicians (no weird fragments)
        final = []
        for b in verified[:10]:
            b2 = b.strip()
            # ensure ends with period if it looks like a sentence
            if len(b2) > 20 and not b2.endswith((".", ";")):
                b2 += "."
            final.append(f"- {b2}")

        return "FINAL:\n" + "\n".join(final)

    def _refine_answer(self, query: str, draft_bullets: List[str], context: str) -> List[str]:
        """v9 Self-Critique: Prunes claims that the LLM realizes are not 100% grounded."""
        draft_text = "\n".join([f"- {b}" for b in draft_bullets])
        prompt = f"""
You are a peer-reviewer for a clinical medical bot.
Your task is to REDUCE HALLUCINATION.

DRAFT ANSWER:
{draft_text}

EVIDENCE:
{context[:2500]}

INSTRUCTION:
1. Examine each bullet point in the DRAFT ANSWER.
2. Cross-reference it with the EVIDENCE.
3. Eliminate or correct any claim that explicitly CONTRADICTS the EVIDENCE. General foundational definitions (like what a disease is) are allowed to remain even if not explicitly supported, so long as they aren't contradicted.
4. Output only the surviving bullet points. 
5. If the draft answer has been entirely rejected, output ABSTAIN.

REFINED OUTPUT:
"""
        raw = _generate_with_prompt(prompt, max_new_tokens=240)
        if "ABSTAIN" in raw.upper() and len(raw) < 15:
            return []
            
        clean = _clean_generated_text(raw, marker="REFINED OUTPUT:")
        return _parse_relaxed_bullets(clean, max_items=12)

    def _generate_patient_answer(self, clinician_text: str) -> str:
        clean = clinician_text.replace("FINAL:", "").strip()

        prompt = f"""
Task: Translate this clinical information into a detailed Patient Summary meant for a 20-year-old to understand.

Rules (strict):
1) Write in clear, detailed paragraphs or bullet points.
2) JARGON: Use medical jargon minimally—only enough to convey meaning without forcing the patient to Google terms. If you use a complex medical term, provide a brief, simple explanation inline.
3) EXPLANATIONS: Provide a fully detailed explanation so the patient completely understands what to expect.

SOURCE TEXT:
{clean}

PATIENT OUTPUT:
""".strip()

        raw = _generate_with_prompt(prompt, max_new_tokens=450)
        patient = _clean_generated_text(raw, marker="PATIENT OUTPUT:")

        return patient

    def _build_abstain_result(self, query: str, decision: Any) -> RAGResult:
        return RAGResult(
            answer="ABSTAIN",
            clinician_answer="ABSTAIN",
            patient_answer="I could not find an explicit answer in the retrieved documents.",
            citations=[],
            confidence=0.0,
            status="abstain",
            source_kbs=[],
            route={"intent": getattr(decision, "intent", "unknown"), "task_hints": getattr(decision, "task_hints", [])},
            debug_info={},
            consistency={"mode": "abstain"},
        )
