"""
mirage_abstention_test_v13.py

Tests ScholarBOT's fail-closed (abstention) behavior using MIRAGE benchmark.
Updated for v13 architecture (11_backend.ScholarBotEngine).

In-domain definition: TB, pneumonia, and related respiratory/pulmonary topics
(broad definition to maximize coverage ~121 questions)

Measures:
  - In-domain answer rate  (should ANSWER)
  - OOD abstention rate    (should ABSTAIN)

Usage:
    cd ~/ScholarBOT
    conda activate is597mlc
    export OPENAI_API_KEY="sk-proj-..."
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    export OMP_NUM_THREADS=1
    python3 mirage_abstention_test_v13.py
"""

import json, os, sys, random, time, importlib
import urllib.request
from pathlib import Path

BENCHMARK_URL   = "https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/refs/heads/main/benchmark.json"
BENCHMARK_CACHE = "mirage_benchmark_cache.json"
N_OOD_SAMPLE    = 50
RANDOM_SEED     = 42

# Broad in-domain keywords: TB + Pneumonia + related respiratory/pulmonary topics
IN_DOMAIN_KEYWORDS = [
    # -- TB core --
    "tuberculosis", "mycobacterium", "isoniazid", "rifampin", "rifampicin",
    "pyrazinamide", "ethambutol", "rifabutin", "bedaquiline", "clofazimine",
    "delamanid", "anti-tuberculosis", "latent tb", "mdr-tb", "tb ",
    # -- TB diagnostics --
    "bcg", "acid-fast", "sputum", "xpert", "dot therapy",
    # -- Pneumonia core --
    "pneumonia", "community-acquired", "streptococcus pneumoniae",
    "pneumococcal", "legionella", "haemophilus influenzae",
    "atypical pneumonia", "mycoplasma", "curb-65",
    # -- Antibiotics used in respiratory infections --
    "azithromycin", "levofloxacin", "doxycycline", "clarithromycin",
    "ceftriaxone", "amoxicillin-clavulanate", "moxifloxacin",
    "ampicillin-sulbactam", "piperacillin", "linezolid",
    # -- Broader respiratory / pulmonary --
    "respiratory tract infection", "lower respiratory", "lung infection",
    "chest infection", "bronchopneumonia", "lobar pneumonia",
    "empyema", "lung abscess", "pleural effusion",
    "pulmonary infiltrate", "consolidation", "cavitation",
    # -- Related pathogens --
    "pseudomonas", "klebsiella pneumoniae",
    # -- Drug classes --
    "macrolide", "fluoroquinolone",
]


def load_benchmark():
    if Path(BENCHMARK_CACHE).exists():
        print(f"Loading cached benchmark from {BENCHMARK_CACHE} ...")
        with open(BENCHMARK_CACHE, encoding="utf-8") as f:
            return json.load(f)
    print("Downloading MIRAGE benchmark.json (may take 1-2 min)...")
    urllib.request.urlretrieve(BENCHMARK_URL, BENCHMARK_CACHE)
    print("Download complete.")
    with open(BENCHMARK_CACHE, encoding="utf-8") as f:
        return json.load(f)


def filter_questions(benchmark):
    in_domain, ood = [], []
    for dataset, questions in benchmark.items():
        for qid, entry in questions.items():
            q_lower = entry.get("question", "").lower()
            is_in   = any(kw in q_lower for kw in IN_DOMAIN_KEYWORDS)
            item = {
                "id":       f"{dataset}_{qid}",
                "dataset":  dataset,
                "question": entry.get("question", ""),
                "domain":   "in_domain" if is_in else "ood",
            }
            (in_domain if is_in else ood).append(item)
    return in_domain, ood


def run_test(engine, questions, label):
    results, answered, abstained = [], 0, 0
    print(f"\nTesting {len(questions)} {label} questions...")
    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {q['dataset']} | {q['question'][:68]}...")
        try:
            response, conf, meta = engine.generate_response(q["question"])
            status     = meta.get("status", "unknown")
            is_abstain = (
                status == "abstain"
                or "ABSTAIN" in response
                or "No confidence" in response
                or conf == 0.0
            )
            decision    = "ABSTAIN" if is_abstain else "ANSWER"
            abstained  += is_abstain
            answered   += not is_abstain
            results.append({
                **q,
                "decision":         decision,
                "confidence":       round(conf, 4),
                "status":           status,
                "response_preview": response[:150],
            })
        except Exception as e:
            abstained += 1
            results.append({
                **q, "decision": "ERROR", "confidence": 0.0,
                "status": "error", "response_preview": str(e),
            })
        time.sleep(0.3)
    return results, answered, abstained


def main():
    print("=" * 60)
    print("MIRAGE Abstention Rate Test — ScholarBOT v13")
    print("=" * 60)

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set!")
        sys.exit(1)

    benchmark             = load_benchmark()
    in_domain, ood        = filter_questions(benchmark)

    print(f"\nQuestion breakdown:")
    print(f"  In-domain (TB + Pneumonia + respiratory) : {len(in_domain)}")
    print(f"  Out-of-domain total                      : {len(ood)}")

    random.seed(RANDOM_SEED)
    ood_sample = random.sample(ood, min(N_OOD_SAMPLE, len(ood)))
    total      = len(in_domain) + len(ood_sample)
    print(f"  OOD sample to test                       : {len(ood_sample)}")
    print(f"  Total questions to run                   : {total}")
    print(f"  Estimated time                           : ~{total * 3 // 60} minutes")

    # ── Load v13 engine ───────────────────────────────────────────────────────
    print("\nLoading ScholarBOT v13 engine...")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    backend = importlib.import_module("11_backend")
    engine  = backend.ScholarBotEngine(
        api_key=os.environ["OPENAI_API_KEY"],
        verbose=False,
    )

    in_res,  in_ans,  in_abs  = run_test(engine, in_domain,  "IN-DOMAIN  (should ANSWER)")
    ood_res, ood_ans, ood_abs = run_test(engine, ood_sample, "OOD        (should ABSTAIN)")

    in_rate  = in_ans  / len(in_domain)  if in_domain  else 0
    ood_rate = ood_abs / len(ood_sample) if ood_sample else 0

    summary = f"""
{'='*60}
MIRAGE ABSTENTION TEST — FINAL RESULTS
{'='*60}

IN-DOMAIN (TB + Pneumonia + respiratory)    {len(in_domain)} questions
  Answered        : {in_ans:3d}   ({in_rate:.1%})
  Abstained       : {in_abs:3d}   ({1-in_rate:.1%})
  ► In-domain Answer Rate : {in_rate:.3f}   (higher = better)

OUT-OF-DOMAIN                               {len(ood_sample)} questions tested
  Abstained       : {ood_abs:3d}   ({ood_rate:.1%})
  Answered        : {ood_ans:3d}   ({1-ood_rate:.1%})
  ► OOD Abstention Rate   : {ood_rate:.3f}   (higher = better)

INTERPRETATION
  High In-domain Answer Rate  → system correctly handles its KB scope
  High OOD Abstention Rate    → fail-closed design working as intended

  The gap between these two rates empirically validates ScholarBOT's
  selective answering behavior — the core fail-closed contribution.
  (In-domain defined as TB, pneumonia, and related respiratory topics)
{'='*60}
"""
    print(summary)

    out = {
        "config": {
            "architecture":        "v13",
            "in_domain_definition": "TB, pneumonia, and related respiratory/pulmonary topics",
            "n_in_domain":         len(in_domain),
            "n_ood_sampled":       len(ood_sample),
            "random_seed":         RANDOM_SEED,
            "keywords_used":       IN_DOMAIN_KEYWORDS,
        },
        "summary": {
            "in_domain_answer_rate":  round(in_rate, 4),
            "ood_abstention_rate":    round(ood_rate, 4),
            "in_domain_total":        len(in_domain),
            "in_domain_answered":     in_ans,
            "in_domain_abstained":    in_abs,
            "ood_total_tested":       len(ood_sample),
            "ood_answered":           ood_ans,
            "ood_abstained":          ood_abs,
        },
        "in_domain_results": in_res,
        "ood_results":       ood_res,
    }

    results_path = os.path.join("eval", "eval results", "mirage_abstention_v13_results.json")
    summary_path = os.path.join("eval", "eval results", "mirage_abstention_v13_summary.txt")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    with open(results_path, "w") as f:
        json.dump(out, f, indent=2)
    with open(summary_path, "w") as f:
        f.write(summary)

    print(f"Saved: {results_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
