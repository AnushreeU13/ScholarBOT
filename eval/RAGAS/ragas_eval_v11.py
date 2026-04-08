
import os
import sys
import json
import logging
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# Set Project Roots
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

# API key validation
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY is required for RAGAS LLM-as-a-Judge.")

# Import v11 Engine
try:
    from aligned_backend import AlignedScholarBotEngine
except ImportError:
    print("Error: Could not import AlignedScholarBotEngine. Ensure you are in the ScholarBOT_v11 directory.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RAGAS_v11")

def load_clinical_dataset(filename):
    # Adjust path to be relative to project root
    full_path = os.path.join(PROJECT_ROOT, filename)
    with open(full_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Using a 10% slice for the first run if desired, or all 200
    questions = [d['question'] for d in data]
    ground_truths = [d['ground_truth'] for d in data]
    return questions, ground_truths

def main():
    logger.info("Initializing ScholarBOT v11 Engine (1024-dim | BGE-Large)...")
    engine = AlignedScholarBotEngine(verbose=False)

    dataset_path = "eval/eval results/eval_dataset_200.json"
    questions, ground_truths = load_clinical_dataset(dataset_path)
    logger.info(f"Loaded {len(questions)} Clinical QA pairs.")

    generated_answers = []
    retrieved_contexts = []

    # Cache handling to avoid re-generating expensive LLM calls
    cache_path = os.path.join(PROJECT_ROOT, "eval/eval results/v11_eval_cache.json")
    
    if os.path.exists(cache_path):
        logger.info(f"Loading results from v11 cache: {cache_path}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
            generated_answers = full_data["generated_answers"]
            retrieved_contexts = full_data["retrieved_contexts"]
    else:
        logger.info("Starting v11 Inference for 200 samples...")
        for i, q in enumerate(questions):
            if i % 10 == 0:
                logger.info(f"Processing clinical query {i}/{len(questions)}...")
            
            try:
                response, conf, meta = engine.generate_response(q)
                # RAGAS expectations: list of strings for context
                evidence = [c.get("text", "") for c in meta.get("evidence_chunks", [])]
                if not evidence: 
                    evidence = ["Fail-Closed: No evidence found."]
                
                generated_answers.append(response)
                retrieved_contexts.append(evidence)
            except Exception as e:
                logger.error(f"Error at query {i}: {e}")
                generated_answers.append("Error: System Timeout")
                retrieved_contexts.append([""])
        
        # Save cache
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({"generated_answers": generated_answers, "retrieved_contexts": retrieved_contexts}, f)

    # Prepare RAGAS Format
    data_dict = {
        "question": questions,
        "answer": generated_answers,
        "contexts": retrieved_contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data_dict)

    logger.info("Starting RAGAS Physician-Grade Evaluation (v11)...")
    
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
    )

    logger.info("\n" + "="*30 + "\n   RAGAS v11 FINAL SCORES\n" + "="*30)
    print(results)

    # Export to CSV for Poster Graphics
    csv_out = os.path.join(PROJECT_ROOT, "eval/RAGAS/ragas_scores_v11.csv")
    results.to_pandas().to_csv(csv_out, index=False)
    logger.info(f"Success! Full results saved to {csv_out}")

if __name__ == "__main__":
    main()
