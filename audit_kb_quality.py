
import json
import time
import os
from pathlib import Path
from typing import List, Dict, Any
import torch

# Configuration
KB_PROCESSED_DIR = Path("datasets/KB_processed")
GUIDELINES_JSONL = KB_PROCESSED_DIR / "guidelines_text" / "guidelines_chunks.jsonl"
DRUGLABELS_JSONL = KB_PROCESSED_DIR / "druglabels_text" / "druglabels_chunks.jsonl"
OUTPUT_REPORT = Path("kb_audit_report.txt")

# Set up your LLM here
import os
from openai import OpenAI
# To run this script locally, set your OPENAI_API_KEY environment variable.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE"))

def audit_chunk(text: str, kb_type: str) -> Dict[str, Any]:
    """Sends a single chunk to the LLM for quality auditing."""
    prompt = f"""
### Task: Clinical & Pharmaceutical Data Audit
You are a Lead Data Quality Engineer for a medical RAG system.
Evaluate the following text chunk from the {kb_type} knowledge base for "Retrieval Noise".

### CHUNK TEXT:
"{text[:2000]}"

### Audit Checklist (Answer exactly with Yes/No):
1. GUIDELINES & DRUG LABELS: Does it contain bibliographies, DOI links, or page headers? (Yes/No)
2. GUIDELINES & DRUG LABELS: Does it contain internal XML tags, DailyMed metadata, or table markers? (Yes/No)
3. FORMATTING: Is there broken text, missing spaces, or confusing punctuation? (Yes/No)
4. NOISE LEVEL (0-10): [Score only]

### Final Recommendation (KEEP / CLEAN / DISCARD):
[Recommendation]

### Analysis:
List the top 2-3 specific issues observed (e.g. "DOI link at end", "Broken footer text", "Reference list residue"). 
If clean, say "None".
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Change to "llama3.1:8b" if using Ollama
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return {"raw": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}

def parse_audit_response(raw_text: str) -> Dict[str, Any]:
    """Parses the LLM output into structured data."""
    results = {
        "has_clutter": "Yes" in raw_text.split("1.")[1].split("\n")[0] if "1." in raw_text else False,
        "has_tags": "Yes" in raw_text.split("2.")[1].split("\n")[0] if "2." in raw_text else False,
        "score": 0,
        "issues": []
    }
    
    # Try to extract score
    try:
        score_match = [int(s) for s in raw_text.split() if s.isdigit() and 0 <= int(s) <= 10]
        if score_match: results["score"] = score_match[0]
    except: pass
    
    return results

def run_audit(file_path: Path, kb_label: str):
    if not file_path.exists():
        print(f"Skipping {kb_label}: File not found at {file_path}")
        return None

    print(f"\n--- Starting Audit for {kb_label} ---")
    all_results = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        total = len(lines)
        
        for i, line in enumerate(lines):
            chunk = json.loads(line)
            text = chunk.get("text", "")
            
            # Progress tracker
            if i % 10 == 0:
                print(f"  Processing {i}/{total}...", end="\r")
            
            audit_data = audit_chunk(text, kb_label)
            if "raw" in audit_data:
                parsed = parse_audit_response(audit_data["raw"])
                all_results.append(parsed)
            
            # Rate limiting safety for API
            # time.sleep(0.1) 

    # Generate Summary
    noise_count = sum(1 for r in all_results if r["score"] > 3)
    avg_score = sum(r["score"] for r in all_results) / len(all_results) if all_results else 0
    noise_pct = (noise_count / len(all_results) * 100) if all_results else 0
    
    report = f"""
{kb_label} Quality Report
---------------------------------
- Total Chunks Audited: {len(all_results)}
- Average Noise Score: {avg_score:.2f}/10
- Percentage of 'Noisy' Chunks (>3 score): {noise_pct:.1f}%

Performance Impact: {'High' if noise_pct > 20 else 'Moderate' if noise_pct > 10 else 'Low'}
"""
    return report

def main():
    final_report = "SCHOLARBOT v10: KNOWLEDGE BASE HEALTH AUDIT\n"
    final_report += "="*50 + "\n"
    
    g_report = run_audit(GUIDELINES_JSONL, "Guidelines")
    if g_report: final_report += g_report
    
    d_report = run_audit(DRUGLABELS_JSONL, "Drug Labels")
    if d_report: final_report += "\n" + d_report
    
    with open(OUTPUT_REPORT, "w") as f:
        f.write(final_report)
    
    print(f"\n\nAudit Complete! Report saved to {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()
