"""
ragas_eval_v13.py
RAGAS evaluation for ScholarBOT v13 architecture.

Run from project root:
    export OPENAI_API_KEY=sk-...
    python eval/RAGAS/ragas_eval_v13.py

Results saved to eval/RAGAS/ragas_scores_v13.csv
"""

import os
import sys
import json
import logging

# ── Path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# ── API key guard ─────────────────────────────────────────────────────────────
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError(
        "OPENAI_API_KEY is not set.\n"
        "Run: export OPENAI_API_KEY=sk-... (Mac/Linux)"
    )

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions    = [d["question"]    for d in data]
    ground_truth = [d["ground_truth"] for d in data]
    return questions, ground_truth


def main():
    logger.info("Initialising ScholarBOT engine (v13 architecture)...")
    import importlib
    backend = importlib.import_module("11_backend")
    engine  = backend.ScholarBotEngine(
        api_key=os.getenv("OPENAI_API_KEY"),
        verbose=False,
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset_path = os.path.join("eval", "eval results", "eval_dataset_v13.json")
    questions, ground_truths = load_dataset(dataset_path)
    logger.info(f"Loaded {len(questions)} evaluation questions.")

    generated_answers: list = []
    retrieved_contexts: list = []

    # ── Cache (allows resuming if interrupted) ────────────────────────────────
    cache_path = os.path.join("eval", "eval results", "v13_eval_full_results_temp.json")

    if os.path.exists(cache_path):
        logger.info(f"Loading cached results from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        generated_answers  = cached["generated_answers"]
        retrieved_contexts = cached["retrieved_contexts"]
    else:
        logger.info("Generating answers for all evaluation questions...")
        for i, q in enumerate(questions):
            if i % 10 == 0:
                logger.info(f"  {i}/{len(questions)}...")
            try:
                response, conf, meta = engine.generate_response(q)
                evidence = [c.get("text", "") for c in meta.get("evidence_chunks", [])]
                if not evidence:
                    evidence = ["No context retrieved."]
                generated_answers.append(response)
                retrieved_contexts.append(evidence)
            except Exception as e:
                logger.error(f"Error on question {i}: {e}")
                generated_answers.append("ABSTAIN (Error)")
                retrieved_contexts.append(["No context retrieved due to error."])

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({
                "generated_answers":  generated_answers,
                "retrieved_contexts": retrieved_contexts,
            }, f)
        logger.info(f"Answers cached to {cache_path}")

    # ── Build RAGAS dataset ───────────────────────────────────────────────────
    dataset = Dataset.from_dict({
        "question":    questions,
        "answer":      generated_answers,
        "contexts":    retrieved_contexts,
        "ground_truth": ground_truths,
    })

    # ── RAGAS evaluate ────────────────────────────────────────────────────────
    logger.info("Running RAGAS evaluation (gpt-4o judge)...")
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        ragas_llm        = ChatOpenAI(model="gpt-4o")
        ragas_embeddings = OpenAIEmbeddings()
    except ImportError:
        logger.warning("langchain_openai not found — using RAGAS defaults.")
        ragas_llm = ragas_embeddings = None

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    logger.info("\n=== RAGAS SCORES (v13 architecture) ===")
    print(results)

    output_path = os.path.join("eval", "RAGAS", "ragas_scores_v13.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_pandas().to_csv(output_path, index=False)
    logger.info(f"Scores saved to {output_path}")


if __name__ == "__main__":
    main()
