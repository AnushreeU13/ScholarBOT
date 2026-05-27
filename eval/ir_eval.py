"""
ir_eval.py
Information Retrieval evaluation for ScholarBOT v13.

Metrics computed at each retrieval stage:
  - NDCG@5, NDCG@10  (Normalised Discounted Cumulative Gain)
  - MRR              (Mean Reciprocal Rank)
  - MAP              (Mean Average Precision)

Stages evaluated:
  1. Dense only      (BGE-large FAISS)
  2. BM25 only       (sparse)
  3. After RRF       (dense + BM25 fused)
  4. After reranker  (cross-encoder on top of RRF)

Relevance judgment (two modes):
  --judge keyword  : chunk is relevant if it contains any key_fact from golden dataset (fast, free)
  --judge llm      : LLM rates each chunk 0/1/2 (slower, costs API calls, more accurate)

Graded relevance:
  keyword mode: 1 if any key_fact present, 0 otherwise
  llm mode:     2 = highly relevant, 1 = partially relevant, 0 = not relevant

Usage:
  python eval/ir_eval.py                      # keyword judge, all questions
  python eval/ir_eval.py --judge llm          # LLM judge
  python eval/ir_eval.py --limit 20           # first 20 questions only
  python eval/ir_eval.py --category clinician # one category only
"""

import argparse
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
GOLDEN_FILE  = SCRIPT_DIR / "golden_dataset.json"
RESULTS_DIR  = SCRIPT_DIR / "eval results"
RESULTS_FILE = RESULTS_DIR / "ir_results.json"

# Must be on sys.path AND cwd for importlib.import_module("01_config") etc. to work
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


# ── Metric implementations ────────────────────────────────────────────────────

def dcg(relevances: List[float], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    total = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        total += rel / math.log2(i + 1)
    return total


def ndcg(relevances: List[float], k: int) -> float:
    """Normalised DCG at k."""
    ideal = sorted(relevances, reverse=True)
    ideal_dcg = dcg(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg(relevances, k) / ideal_dcg


def reciprocal_rank(relevances: List[float]) -> float:
    """Reciprocal rank of the first relevant document (rel > 0)."""
    for i, rel in enumerate(relevances, start=1):
        if rel > 0:
            return 1.0 / i
    return 0.0


def average_precision(relevances: List[float]) -> float:
    """Average precision over the ranked list."""
    hits, total_ap = 0, 0.0
    n_relevant = sum(1 for r in relevances if r > 0)
    if n_relevant == 0:
        return 0.0
    for i, rel in enumerate(relevances, start=1):
        if rel > 0:
            hits += 1
            total_ap += hits / i
    return total_ap / n_relevant


def aggregate(metric_lists: List[List[float]], k5=5, k10=10) -> Dict:
    """Compute mean NDCG@5, NDCG@10, MRR, MAP over all queries."""
    n = len(metric_lists)
    if n == 0:
        return {"ndcg@5": 0.0, "ndcg@10": 0.0, "mrr": 0.0, "map": 0.0, "n_queries": 0}
    return {
        "ndcg@5":    round(sum(ndcg(r, k5)             for r in metric_lists) / n, 4),
        "ndcg@10":   round(sum(ndcg(r, k10)            for r in metric_lists) / n, 4),
        "mrr":       round(sum(reciprocal_rank(r)       for r in metric_lists) / n, 4),
        "map":       round(sum(average_precision(r)     for r in metric_lists) / n, 4),
        "n_queries": n,
    }


# ── Relevance judges ──────────────────────────────────────────────────────────

def _keyword_relevance(chunk_text: str, key_facts: List[str]) -> float:
    """Binary relevance: 1 if any key_fact appears in chunk text, else 0."""
    text = chunk_text.lower()
    for fact in key_facts:
        if fact.lower() in text:
            return 1.0
    return 0.0


_LLM_JUDGE_SYSTEM = (
    "You are an information retrieval judge. "
    "Rate how relevant a text chunk is to answering the given question. "
    "Output ONLY a single integer: 2 (highly relevant — directly answers the question), "
    "1 (partially relevant — contains related information), "
    "or 0 (not relevant)."
)


def _llm_relevance(chunk_text: str, question: str, model: str = "gpt-4o-mini") -> float:
    """Graded relevance 0/1/2 from LLM judge."""
    prompt = (
        f"Question: {question}\n\n"
        f"Text chunk:\n{chunk_text[:600]}\n\n"
        f"Relevance (0, 1, or 2):"
    )
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _LLM_JUDGE_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,
            max_tokens=5,
        )
        val = (resp.choices[0].message.content or "0").strip()
        score = float(val[0]) if val and val[0] in "012" else 0.0
        return min(score, 2.0)
    except Exception as e:
        print(f"  [Judge] LLM error: {e}")
        return 0.0


# ── Stage-level retrieval ─────────────────────────────────────────────────────

def _retrieve_stages(retriever, query: str, target_kbs: List[str]) -> Dict[str, List[Dict]]:
    """
    Run retrieval at four stages and return ranked chunk lists for each.
    Returns: {stage_name: [chunk_dict, ...]}
    """
    import re
    _cfg = importlib.import_module("01_config")
    _ret = importlib.import_module("09_retriever")

    # ── Dense only ────────────────────────────────────────────────────────────
    dense_results = []
    for kb in target_kbs:
        dense_results.extend(retriever._dense_search(query, kb, _cfg.TOP_K_DENSE))
    dense_results.sort(key=lambda x: x["score"], reverse=True)

    # ── BM25 only ─────────────────────────────────────────────────────────────
    sparse_results = []
    for kb in target_kbs:
        sparse_results.extend(retriever._bm25_search(query, kb, _cfg.TOP_K_SPARSE))
    sparse_results.sort(key=lambda x: x["score"], reverse=True)

    # ── After RRF ─────────────────────────────────────────────────────────────
    rrf_results = []
    for kb in target_kbs:
        d = retriever._dense_search(query, kb, _cfg.TOP_K_DENSE)
        s = retriever._bm25_search(query, kb, _cfg.TOP_K_SPARSE)
        merged = _ret._rrf_merge(d, s) if s else d
        rrf_results.extend(merged)

    seen, rrf_unique = set(), []
    for c in rrf_results:
        key = re.sub(r"\W+", "", c["text"][:100]).lower()
        if key and key not in seen:
            seen.add(key)
            rrf_unique.append(c)
    rrf_unique.sort(key=lambda x: x["score"], reverse=True)

    # ── After reranker ────────────────────────────────────────────────────────
    reranked = rrf_unique[:_cfg.TOP_K_DENSE]
    if reranked:
        try:
            reranker = _ret._get_reranker()
            pairs  = [[query, c["text"]] for c in reranked]
            scores = reranker.predict(pairs)
            for c, s in zip(reranked, scores):
                c["rerank_score"] = float(s)
            reranked = sorted(reranked, key=lambda x: x.get("rerank_score", 0), reverse=True)
            reranked = reranked[:_cfg.RERANK_K]
        except Exception as e:
            print(f"  [IR] Reranking failed: {e}")
            reranked = reranked[:_cfg.RERANK_K]

    return {
        "dense":    dense_results[:20],
        "bm25":     sparse_results[:20],
        "rrf":      rrf_unique[:20],
        "reranker": reranked,
    }


# ── Load engine components ────────────────────────────────────────────────────

def load_retriever():
    """Load ScholarBOT's HybridRetriever and route the two main KBs."""
    _backend = importlib.import_module("11_backend")
    engine = _backend.ScholarBotEngine(api_key=os.getenv("OPENAI_API_KEY", ""))

    # Extract the retriever from the engine
    retriever = engine.retriever
    return retriever, engine


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="IR evaluation for ScholarBOT v13.")
    ap.add_argument("--judge",    choices=["keyword", "llm"], default="keyword",
                    help="Relevance judge: keyword (fast) or llm (accurate)")
    ap.add_argument("--limit",    type=int, default=None,
                    help="Max questions to evaluate (default: all in-domain)")
    ap.add_argument("--category", choices=["clinician", "patient", "all"], default="all",
                    help="Which question category to evaluate (default: all)")
    args = ap.parse_args()

    if args.judge == "llm" and not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY required for LLM judge.")
        sys.exit(1)

    # ── Load golden dataset ───────────────────────────────────────────────────
    with open(GOLDEN_FILE, encoding="utf-8") as f:
        golden = json.load(f)

    questions = [
        q for q in golden["questions"]
        if q["expected_behaviour"] == "answer"
        and (args.category == "all" or q["category"] == args.category)
    ]
    if args.limit:
        questions = questions[:args.limit]

    print(f"[IR Eval] {len(questions)} questions | judge={args.judge} | category={args.category}")

    # ── Load retriever ────────────────────────────────────────────────────────
    print("[IR Eval] Loading ScholarBOT engine...")
    retriever, engine = load_retriever()
    target_kbs = ["guidelines_kb", "druglabels_kb"]
    print("  Ready.\n")

    # ── Per-stage relevance lists ─────────────────────────────────────────────
    stage_relevances: Dict[str, List[List[float]]] = {
        "dense":    [],
        "bm25":     [],
        "rrf":      [],
        "reranker": [],
    }

    per_question_results = []

    for qi, q in enumerate(questions):
        qid      = q["id"]
        question = q["question"]
        key_facts = q.get("key_facts", [])

        print(f"[{qi+1:03d}/{len(questions)}] {qid} — {question[:70]}...")

        # Retrieve at all stages
        stages = _retrieve_stages(retriever, question, target_kbs)

        q_result = {"id": qid, "question": question, "stages": {}}

        for stage_name, chunks in stages.items():
            relevances = []
            for chunk in chunks:
                text = chunk.get("text", "")
                if args.judge == "llm":
                    rel = _llm_relevance(text, question)
                    time.sleep(0.2)
                else:
                    rel = _keyword_relevance(text, key_facts)
                relevances.append(rel)

            stage_relevances[stage_name].append(relevances)

            q_result["stages"][stage_name] = {
                "ndcg@5":  round(ndcg(relevances, 5),  4),
                "ndcg@10": round(ndcg(relevances, 10), 4),
                "mrr":     round(reciprocal_rank(relevances), 4),
                "ap":      round(average_precision(relevances), 4),
                "n_chunks_retrieved": len(chunks),
                "n_relevant": sum(1 for r in relevances if r > 0),
            }

        per_question_results.append(q_result)

    # ── Aggregate scores ──────────────────────────────────────────────────────
    summary = {}
    for stage_name, rel_lists in stage_relevances.items():
        summary[stage_name] = aggregate(rel_lists)

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"IR EVALUATION — ScholarBOT v13  |  judge={args.judge}")
    print("=" * 65)
    print(f"{'Stage':<14} {'NDCG@5':>8} {'NDCG@10':>9} {'MRR':>8} {'MAP':>8}  {'N':>5}")
    print("-" * 65)
    for stage in ["dense", "bm25", "rrf", "reranker"]:
        s = summary[stage]
        print(f"{stage:<14} {s['ndcg@5']:>8.4f} {s['ndcg@10']:>9.4f} "
              f"{s['mrr']:>8.4f} {s['map']:>8.4f}  {s['n_queries']:>5}")
    print("=" * 65)

    # ── Save ──────────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "config": {
            "judge":    args.judge,
            "category": args.category,
            "n_questions": len(questions),
        },
        "summary": summary,
        "per_question": per_question_results,
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
