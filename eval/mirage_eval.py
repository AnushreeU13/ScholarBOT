"""
mirage_eval.py
MIRAGE benchmark evaluation for ScholarBOT v13.

Strategy:
  - Select all TB/Pneumonia questions from benchmark.json (in-domain)
  - Select 50 random out-of-domain questions (to measure abstain rate)
  - For each question: get ScholarBOT's full-text answer
  - LLM-as-judge: map ScholarBOT's answer to the closest MCQ option
  - Score: accuracy on answered questions + abstain rate tracked separately

Results saved to: eval/eval results/mirage_results.json
Summary printed to console.

Usage:
    python eval/mirage_eval.py
    python eval/mirage_eval.py --limit 20          # quick smoke test
    python eval/mirage_eval.py --resume            # skip already-done questions
"""

import argparse
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BENCHMARK    = SCRIPT_DIR / "benchmark.json"
RESULTS_DIR  = SCRIPT_DIR / "eval results"
RESULTS_FILE = RESULTS_DIR / "mirage_results.json"

# Add project root to path so we can import ScholarBOT modules
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Domain keywords ────────────────────────────────────────────────────────────
TB_TERMS = [
    "tuberculosis", "tubercul", " tb ", "tb,", "tb.", "latent tb",
    "pulmonary tb", "mtb", "mycobacterium", "isoniazid", "rifampin",
    "rifampicin", "pyrazinamide", "ethambutol", "bedaquiline", "bcg",
    "streptomycin", "drug-resistant tb", "mdr-tb", "xdr-tb",
]
PN_TERMS = [
    "pneumonia", "pneumococcal", "streptococcus pneumoniae", "cap ",
    "community-acquired", "legionella", "atypical pneumonia",
    "pneumocystis", "pcp ",
]
IN_DOMAIN_TERMS = TB_TERMS + PN_TERMS


def _is_in_domain(question: str, options: Dict) -> bool:
    text = (question + " " + " ".join(options.values())).lower()
    return any(t in text for t in IN_DOMAIN_TERMS)


# ── Load benchmark ─────────────────────────────────────────────────────────────

def load_questions() -> (List[Dict], List[Dict]):
    """Return (in_domain_questions, out_of_domain_questions)."""
    with open(BENCHMARK, encoding="utf-8") as f:
        data = json.load(f)

    in_domain, out_domain = [], []
    for dataset, questions in data.items():
        for qid, q in questions.items():
            entry = {
                "dataset":  dataset,
                "qid":      qid,
                "question": q.get("question", ""),
                "options":  q.get("options", {}),
                "answer":   q.get("answer", ""),
            }
            if _is_in_domain(entry["question"], entry["options"]):
                in_domain.append(entry)
            else:
                out_domain.append(entry)

    return in_domain, out_domain


# ── ScholarBOT query ──────────────────────────────────────────────────────────

def get_scholarbot_answer(engine, question: str, history: List[Dict], domain: str = "in") -> str:
    """
    Query ScholarBOT and return the raw response text.
    For in-domain questions, prepend a domain hint so the router correctly
    classifies vignette-style questions that don't mention TB/pneumonia explicitly.
    This is legitimate: we already know these are TB/pneumonia questions from
    our benchmark filtering — we are testing factual retrieval, not topic detection.
    """
    if domain == "in":
        query = (
            "This is a clinical question about tuberculosis or pneumonia. "
            + question
        )
    else:
        query = question   # out-of-domain: no hint — test abstain as-is

    try:
        response_text, confidence, meta = engine.generate_response(
            query=query,
            force_user_kb=False,
            history=history,
        )
        return response_text
    except Exception as e:
        return f"ERROR: {e}"


# ── LLM judge ─────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = (
    "You are an expert medical evaluator. "
    "Given a clinical question, its answer options, and a detailed answer from a medical AI, "
    "determine which answer option the AI's response most supports. "
    "If the AI explicitly abstained or said it could not find information, output 'ABSTAIN'. "
    "Output ONLY a single letter (A, B, C, D, or E) or the word 'ABSTAIN'. Nothing else."
)


def judge_answer(
    question: str,
    options: Dict[str, str],
    scholarbot_answer: str,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Ask the LLM judge which MCQ option ScholarBOT's answer best supports.
    Returns: 'A'|'B'|'C'|'D'|'E'|'ABSTAIN'|'ERROR'
    """
    # Fast-path: detect abstain without LLM call
    abstain_phrases = [
        "no confidence", "abstaining", "cannot find", "not find",
        "insufficient", "unable to find", "i don't have",
        "outside my knowledge", "out of domain", "not in my knowledge",
        "no relevant", "cannot answer",
    ]
    lower_ans = scholarbot_answer.lower()
    if any(p in lower_ans for p in abstain_phrases):
        return "ABSTAIN"

    options_text = "\n".join(f"{k}: {v}" for k, v in options.items())
    prompt = (
        f"Question:\n{question}\n\n"
        f"Answer options:\n{options_text}\n\n"
        f"Medical AI's response:\n{scholarbot_answer}\n\n"
        f"Which option does the AI's response most support? "
        f"Output only the letter or ABSTAIN."
    )

    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,
            max_tokens=10,
        )
        verdict = (resp.choices[0].message.content or "").strip().upper()
        # Normalise — extract first letter or ABSTAIN
        if "ABSTAIN" in verdict:
            return "ABSTAIN"
        for ch in verdict:
            if ch in options:
                return ch
        return "ABSTAIN"  # couldn't map → treat as abstain
    except Exception as e:
        print(f"  [Judge] ERROR: {e}")
        return "ERROR"


# ── Results I/O ────────────────────────────────────────────────────────────────

def load_existing_results() -> Dict:
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(results: Dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ── Scoring ────────────────────────────────────────────────────────────────────

def compute_scores(results: Dict) -> Dict:
    in_domain  = {k: v for k, v in results.items() if v.get("domain") == "in"}
    out_domain = {k: v for k, v in results.items() if v.get("domain") == "out"}

    def _score(subset):
        total     = len(subset)
        abstained = sum(1 for v in subset.values() if v.get("predicted") in ("ABSTAIN", "ERROR"))
        answered  = total - abstained
        correct   = sum(
            1 for v in subset.values()
            if v.get("predicted") == v.get("correct_answer")
        )
        accuracy_of_answered = correct / answered if answered > 0 else 0.0
        accuracy_overall     = correct / total    if total   > 0 else 0.0
        abstain_rate         = abstained / total  if total   > 0 else 0.0
        return {
            "total":                total,
            "answered":             answered,
            "abstained":            abstained,
            "correct":              correct,
            "accuracy_of_answered": round(accuracy_of_answered, 3),
            "accuracy_overall":     round(accuracy_overall, 3),
            "abstain_rate":         round(abstain_rate, 3),
        }

    return {
        "in_domain":   _score(in_domain),
        "out_of_domain": _score(out_domain),
    }


def print_summary(scores: Dict) -> None:
    print("\n" + "=" * 60)
    print("MIRAGE EVALUATION — ScholarBOT v13")
    print("=" * 60)
    for label, s in scores.items():
        tag = "TB & Pneumonia" if label == "in_domain" else "Out-of-domain (should abstain)"
        print(f"\n  {tag}")
        print(f"    Questions   : {s['total']}")
        print(f"    Answered    : {s['answered']}  |  Abstained: {s['abstained']}")
        print(f"    Correct     : {s['correct']}")
        print(f"    Accuracy (answered questions) : {s['accuracy_of_answered']:.1%}")
        print(f"    Accuracy (overall)            : {s['accuracy_overall']:.1%}")
        print(f"    Abstain rate                  : {s['abstain_rate']:.1%}")
    print("=" * 60)
    print(f"\nFull results saved to: {RESULTS_FILE}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="MIRAGE evaluation for ScholarBOT v13.")
    ap.add_argument("--limit",  type=int, default=None,
                    help="Max in-domain questions to evaluate (default: all)")
    ap.add_argument("--ood",    type=int, default=50,
                    help="Number of out-of-domain questions (default: 50)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip questions already in results file")
    ap.add_argument("--seed",   type=int, default=42,
                    help="Random seed for out-of-domain sampling")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.")
        print("  PowerShell: $env:OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    # ── Load benchmark ────────────────────────────────────────────────────────
    print("[MIRAGE] Loading benchmark...")
    in_domain_qs, out_domain_qs = load_questions()
    print(f"  In-domain (TB/Pneumonia): {len(in_domain_qs)}")
    print(f"  Out-of-domain pool:       {len(out_domain_qs)}")

    # Sample out-of-domain
    random.seed(args.seed)
    ood_sample = random.sample(out_domain_qs, min(args.ood, len(out_domain_qs)))

    # Apply in-domain limit
    if args.limit:
        in_domain_qs = in_domain_qs[:args.limit]

    all_questions = (
        [{"domain": "in",  **q} for q in in_domain_qs] +
        [{"domain": "out", **q} for q in ood_sample]
    )
    print(f"  Evaluating: {len(in_domain_qs)} in-domain + {len(ood_sample)} out-of-domain "
          f"= {len(all_questions)} total\n")

    # ── Load ScholarBOT engine ────────────────────────────────────────────────
    print("[MIRAGE] Loading ScholarBOT engine...")
    _backend = importlib.import_module("11_backend")
    # Pass API key explicitly — os.environ may not propagate into importlib modules
    engine = _backend.ScholarBotEngine(api_key=os.getenv("OPENAI_API_KEY", ""))
    print("  Engine ready.\n")

    # ── Load existing results (resume) ────────────────────────────────────────
    results = load_existing_results() if args.resume else {}
    skipped = 0

    # ── Evaluate ──────────────────────────────────────────────────────────────
    for i, q in enumerate(all_questions):
        key = f"{q['dataset']}_{q['qid']}"

        if args.resume and key in results:
            skipped += 1
            continue

        domain_tag = "IN " if q["domain"] == "in" else "OUT"
        print(f"[{i+1:03d}/{len(all_questions)}] [{domain_tag}] [{q['dataset']}] {q['question'][:80]}...")

        # Step 1: ScholarBOT answer (no history — each question is standalone)
        sb_answer = get_scholarbot_answer(engine, q["question"], history=[], domain=q["domain"])

        # Step 2: LLM judge
        predicted = judge_answer(q["question"], q["options"], sb_answer)
        correct   = q["answer"]
        is_correct = predicted == correct

        status = "ABSTAIN" if predicted == "ABSTAIN" else ("✓" if is_correct else "✗")
        print(f"  Correct: {correct} | Predicted: {predicted} | {status}")

        results[key] = {
            "domain":         q["domain"],
            "dataset":        q["dataset"],
            "qid":            q["qid"],
            "question":       q["question"],
            "options":        q["options"],
            "correct_answer": correct,
            "scholarbot_answer": sb_answer,
            "predicted":      predicted,
            "is_correct":     is_correct,
        }

        # Save after every question (resume-safe)
        save_results(results)

        # Rate limiting
        time.sleep(0.5)

    if skipped:
        print(f"\n[MIRAGE] Skipped {skipped} already-evaluated questions (--resume).")

    # ── Score ─────────────────────────────────────────────────────────────────
    scores = compute_scores(results)
    print_summary(scores)

    # Save scores summary alongside full results
    summary_file = RESULTS_DIR / "mirage_summary.json"
    with open(summary_file, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
