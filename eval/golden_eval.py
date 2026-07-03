"""
golden_eval.py
End-to-end evaluation of ScholarBOT v13 against the purpose-built golden dataset.

Metrics:
  answer questions  — key_facts recall: fraction of key_facts present in response
  abstain questions — abstain rate: fraction correctly refused (should be 1.0)
  faithfulness      — fraction of answer responses containing citation markers [N]

Reported per category: clinician_tb, clinician_pn, patient_tb, patient_pn, ood

Usage:
    python eval/golden_eval.py
    python eval/golden_eval.py --limit 20
    python eval/golden_eval.py --category clinician
    python eval/golden_eval.py --resume
"""

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
GOLDEN_FILE  = SCRIPT_DIR / "golden_dataset.json"
RESULTS_DIR  = SCRIPT_DIR / "eval results"
RESULTS_FILE = RESULTS_DIR / "golden_results.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


# ── Scoring helpers ───────────────────────────────────────────────────────────

def key_facts_recall(response: str, key_facts: List[str]) -> float:
    """Fraction of key_facts whose text appears in the response (case-insensitive)."""
    if not key_facts:
        return 1.0
    text = response.lower()
    found = sum(1 for f in key_facts if f.lower() in text)
    return round(found / len(key_facts), 4)


def is_abstain(response: str) -> bool:
    """True if ScholarBOT declined to answer."""
    markers = [
        "no confidence", "abstaining", "cannot find", "not find",
        "insufficient", "unable to find", "i don't have",
        "outside my knowledge", "out of domain", "not in my knowledge",
        "no relevant", "cannot answer", "outside scholarbot",
    ]
    lower = response.lower()
    return any(m in lower for m in markers)


def has_citations(response: str) -> bool:
    """True if response contains at least one inline citation like [1]."""
    import re
    return bool(re.search(r"\[\d+\]", response))


# ── Category label ────────────────────────────────────────────────────────────

def _category_key(q: Dict) -> str:
    cat   = q.get("category", "")
    topic = q.get("topic", "").lower()
    if "tb" in topic or "tuberculosis" in topic:
        return f"{cat}_tb"
    if "pneumonia" in topic or "pn" in topic:
        return f"{cat}_pn"
    if cat == "ood":
        return "ood"
    return cat


# ── Aggregate ─────────────────────────────────────────────────────────────────

def aggregate(per_question: List[Dict]) -> Dict:
    """Compute overall and per-category metrics from per-question results."""
    answer_qs  = [r for r in per_question if r["expected"] == "answer"]
    abstain_qs = [r for r in per_question if r["expected"] == "abstain"]

    # Overall answer metrics
    if answer_qs:
        mean_recall = round(sum(r["recall"] for r in answer_qs) / len(answer_qs), 4)
        faithfulness = round(sum(1 for r in answer_qs if r["has_citations"]) / len(answer_qs), 4)
        wrongly_abstained = sum(1 for r in answer_qs if r["abstained"])
    else:
        mean_recall = faithfulness = 0.0
        wrongly_abstained = 0

    # Abstain calibration
    if abstain_qs:
        abstain_rate = round(sum(1 for r in abstain_qs if r["abstained"]) / len(abstain_qs), 4)
    else:
        abstain_rate = None

    # Per-category breakdown
    categories: Dict[str, Dict] = {}
    for r in per_question:
        cat = r["category_key"]
        if cat not in categories:
            categories[cat] = {"recall": [], "abstain_correct": [], "total": 0}
        categories[cat]["total"] += 1
        if r["expected"] == "answer":
            categories[cat]["recall"].append(r["recall"])
        else:
            categories[cat]["abstain_correct"].append(int(r["abstained"]))

    cat_summary = {}
    for cat, data in categories.items():
        cat_summary[cat] = {
            "total":          data["total"],
            "mean_recall":    round(sum(data["recall"]) / len(data["recall"]), 4) if data["recall"] else None,
            "abstain_rate":   round(sum(data["abstain_correct"]) / len(data["abstain_correct"]), 4)
                              if data["abstain_correct"] else None,
        }

    return {
        "n_answer_questions":  len(answer_qs),
        "n_abstain_questions": len(abstain_qs),
        "mean_key_facts_recall": mean_recall,
        "faithfulness":          faithfulness,
        "wrongly_abstained":     wrongly_abstained,
        "abstain_calibration":   abstain_rate,
        "per_category":          cat_summary,
    }


# ── Print summary ─────────────────────────────────────────────────────────────

def print_summary(summary: Dict) -> None:
    print("\n" + "=" * 65)
    print("GOLDEN DATASET EVALUATION — ScholarBOT v13")
    print("=" * 65)
    print(f"  Answer questions    : {summary['n_answer_questions']}")
    print(f"  Key-facts recall    : {summary['mean_key_facts_recall']:.1%}  "
          f"(fraction of key facts present in answer)")
    print(f"  Faithfulness        : {summary['faithfulness']:.1%}  "
          f"(answers with citation markers)")
    print(f"  Wrongly abstained   : {summary['wrongly_abstained']}  "
          f"(answerable questions refused)")
    if summary["abstain_calibration"] is not None:
        print(f"\n  OOD abstain rate    : {summary['abstain_calibration']:.1%}  "
              f"(should be 1.0)")

    print(f"\n  {'Category':<20} {'N':>4} {'Recall':>8} {'Abstain':>9}")
    print("  " + "-" * 45)
    for cat, s in summary["per_category"].items():
        recall  = f"{s['mean_recall']:.1%}"  if s["mean_recall"]  is not None else "   —"
        abstain = f"{s['abstain_rate']:.1%}" if s["abstain_rate"] is not None else "   —"
        print(f"  {cat:<20} {s['total']:>4} {recall:>8} {abstain:>9}")
    print("=" * 65)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Golden dataset evaluation for ScholarBOT v13.")
    ap.add_argument("--limit",    type=int, default=None,
                    help="Max questions to evaluate (default: all)")
    ap.add_argument("--category", choices=["clinician", "patient", "ood", "all"], default="all",
                    help="Filter by question category")
    ap.add_argument("--resume",   action="store_true",
                    help="Skip questions already in results file")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.")
        print("  PowerShell: $env:OPENAI_API_KEY='sk-proj-...'")
        sys.exit(1)

    # ── Load golden dataset ───────────────────────────────────────────────────
    with open(GOLDEN_FILE, encoding="utf-8") as f:
        golden = json.load(f)

    questions = golden["questions"]
    if args.category != "all":
        questions = [q for q in questions if q.get("category") == args.category]
    if args.limit:
        questions = questions[:args.limit]

    print(f"[GoldenEval] {len(questions)} questions | category={args.category}")

    # ── Load engine ───────────────────────────────────────────────────────────
    print("[GoldenEval] Loading ScholarBOT engine...")
    _backend = importlib.import_module("11_backend")
    engine   = _backend.ScholarBotEngine(api_key=os.getenv("OPENAI_API_KEY", ""))
    print("  Engine ready.\n")

    # ── Load existing results if resuming ────────────────────────────────────
    per_question: List[Dict] = []
    done_ids: set = set()

    if args.resume and RESULTS_FILE.exists():
        with open(RESULTS_FILE, encoding="utf-8") as f:
            existing = json.load(f)
        per_question = existing.get("per_question", [])
        done_ids = {r["id"] for r in per_question}
        print(f"[GoldenEval] Resuming — {len(done_ids)} already done.\n")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    for qi, q in enumerate(questions):
        qid      = q["id"]
        question = q["question"]
        expected = q.get("expected_behaviour", "answer")
        key_facts = q.get("key_facts", [])

        if qid in done_ids:
            continue

        cat_key = _category_key(q)
        print(f"[{qi+1:03d}/{len(questions)}] {qid} [{cat_key}] — {question[:70]}...")

        try:
            response_text, confidence, meta = engine.generate_response(
                query=question,
                force_user_kb=False,
                history=[],
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            response_text = f"ERROR: {e}"
            confidence    = 0.0
            meta          = {}

        abstained   = is_abstain(response_text)
        recall      = 0.0 if abstained else key_facts_recall(response_text, key_facts)
        cited       = has_citations(response_text) if not abstained else False

        status_icon = (
            "ABSTAIN" if abstained
            else f"recall={recall:.0%}"
        )
        print(f"  → {status_icon}  conf={confidence:.3f}")

        per_question.append({
            "id":           qid,
            "category_key": cat_key,
            "question":     question,
            "expected":     expected,
            "abstained":    abstained,
            "recall":       recall,
            "has_citations": cited,
            "confidence":   round(confidence, 4),
            "response":     response_text[:600],
            "key_facts_found": [f for f in key_facts if f.lower() in response_text.lower()],
            "key_facts_missing": [f for f in key_facts if f.lower() not in response_text.lower()],
        })

        # Save after every question — resume-safe
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        _interim = {
            "config":       {"category": args.category, "n_questions": len(questions)},
            "summary":      aggregate(per_question),
            "per_question": per_question,
        }
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(_interim, f, indent=2, ensure_ascii=False)

        time.sleep(0.3)

    # ── Final summary ─────────────────────────────────────────────────────────
    summary = aggregate(per_question)
    print_summary(summary)

    output = {
        "config":       {"category": args.category, "n_questions": len(questions)},
        "summary":      summary,
        "per_question": per_question,
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
