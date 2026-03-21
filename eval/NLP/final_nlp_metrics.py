
import json
import re
import numpy as np
import pandas as pd
from typing import List
from rouge_score import rouge_scorer

def load_dataset(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    questions = [d['question'] for d in data]
    ground_truths = [d['ground_truth'] for d in data]
    return questions, ground_truths

def get_tokens(text: str) -> List[str]:
    if not text: return []
    # Simple tokenization: alphanumeric only
    return re.findall(r'\w+', text.lower())

def calculate_metrics(prediction: str, ground_truth: str) -> dict:
    pred_tokens = get_tokens(prediction)
    gt_tokens = get_tokens(ground_truth)
    
    if not gt_tokens:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    if not pred_tokens:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
        
    common = set(pred_tokens) & set(gt_tokens)
    num_same = len(common)
    
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"f1": f1, "precision": precision, "recall": recall}

def clean_answer(text: str) -> str:
    if "No confidence" in text:
        return ""
    res = text
    if "### Evidence" in res:
        res = res.split("### Evidence")[0]
    # Keep the summaries
    return res.strip()

def main():
    with open("v8_eval_full_results.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    generated_answers = data["generated_answers"]
    
    _, ground_truths = load_dataset("../../eval_dataset_200.json")

    f1_list = []
    rec_list = []
    prec_list = []
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    r1_list = []
    rl_list = []

    for pred_raw, gt in zip(generated_answers, ground_truths):
        pred = clean_answer(pred_raw)
        m = calculate_metrics(pred, gt)
        f1_list.append(m["f1"])
        rec_list.append(m["recall"])
        prec_list.append(m["precision"])
        
        scores = scorer.score(gt, pred if pred else "ABSTAIN")
        r1_list.append(scores['rouge1'].fmeasure)
        rl_list.append(scores['rougeL'].fmeasure)

    print(f"Token F1: {np.mean(f1_list):.4f}")
    print(f"Token Recall: {np.mean(rec_list):.4f}")
    print(f"Token Precision: {np.mean(prec_list):.4f}")
    print(f"ROUGE-1:  {np.mean(r1_list):.4f}")
    print(f"ROUGE-L:  {np.mean(rl_list):.4f}")

    # Final summary for walkthrough
    # We'll use Token F1 and ROUGE-L as requested.
    results = {
        "Token F1": np.mean(f1_list),
        "Token Recall": np.mean(rec_list),
        "ROUGE-L": np.mean(rl_list)
    }
    with open("nlp_metrics_v8_final.json", 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
