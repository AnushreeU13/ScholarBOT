"""
generate_eval_dataset_v13.py
Generate 200 evaluation QA pairs from the expanded ScholarBOT KB.
Samples chunks from guidelines_kb (3225 records) and druglabels_kb (15185 records),
calls GPT-4o-mini to generate question + ground_truth answer pairs.

Run from project root:
    export OPENAI_API_KEY=sk-...
    python eval/generate_eval_dataset_v13.py

Output: eval/eval results/eval_dataset_v13.json
Format: [{"question": "...", "ground_truth": "..."}, ...]
"""

import os
import sys
import json
import random
import time
import logging
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
DATASET_DIR    = PROJECT_ROOT / "dataset"
OUTPUT_DIR     = PROJECT_ROOT / "eval" / "eval results"
OUTPUT_FILE    = OUTPUT_DIR / "eval_dataset_v13.json"
CACHE_FILE     = OUTPUT_DIR / "eval_dataset_v13_cache.json"

GUIDELINES_FILE   = DATASET_DIR / "guidelines_chunks_cleaned.jsonl"
CDC_TB_FILE       = DATASET_DIR / "cdc_tb_pages.jsonl"
DRUGLABELS_FILE   = DATASET_DIR / "druglabels_chunks.jsonl"

TOTAL_QUESTIONS   = 200
GUIDELINES_N      = 130   # ~65% from guidelines (clinical knowledge)
DRUGLABELS_N      = 70    # ~35% from drug labels (pharmacology)

MIN_CHUNK_LEN     = 150   # skip very short chunks
MAX_CHUNK_LEN     = 2000  # skip very long chunks (truncate context)
RETRY_LIMIT       = 3
SLEEP_BETWEEN     = 0.5   # seconds between API calls

# ── API key guard ─────────────────────────────────────────────────────────────
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError(
        "OPENAI_API_KEY is not set.\n"
        "Run: export OPENAI_API_KEY=sk-..."
    )

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def get_text(record: dict) -> str:
    return record.get("text") or record.get("chunk_text") or ""


def filter_chunks(records: list[dict]) -> list[dict]:
    """Keep chunks with useful length."""
    return [
        r for r in records
        if MIN_CHUNK_LEN <= len(get_text(r)) <= MAX_CHUNK_LEN
    ]


def generate_qa(chunk_text: str, source_hint: str) -> dict | None:
    """Call GPT-4o-mini to generate one question + ground_truth from a chunk."""
    prompt = f"""You are a clinical knowledge evaluator for a medical RAG system focused on tuberculosis (TB) and community-acquired pneumonia (CAP).

Given the following clinical text excerpt, generate ONE high-quality evaluation question and a comprehensive ground-truth answer.

Requirements:
- The question must be answerable using ONLY the provided text
- The question should test clinical understanding (not trivial lookup)
- The ground truth answer must be 2-5 complete sentences, clinically accurate, and written in professional medical language
- Do NOT ask about page numbers, document titles, or metadata
- Focus on: diagnosis, treatment, drug dosing, monitoring, resistance, or management guidelines

Source: {source_hint}

Text:
{chunk_text[:1500]}

Respond with valid JSON only, no markdown:
{{"question": "...", "ground_truth": "..."}}"""

    for attempt in range(RETRY_LIMIT):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400,
            )
            content = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            parsed = json.loads(content)
            if "question" in parsed and "ground_truth" in parsed:
                return parsed
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Parse error attempt {attempt+1}: {e}")
            time.sleep(1)
        except Exception as e:
            logger.warning(f"API error attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    return None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load cache if exists (allows resuming)
    qa_pairs: list[dict] = []
    if CACHE_FILE.exists():
        with open(CACHE_FILE, encoding="utf-8") as f:
            qa_pairs = json.load(f)
        logger.info(f"Resuming from cache: {len(qa_pairs)} pairs already generated.")

    already_done = len(qa_pairs)
    guidelines_done  = sum(1 for q in qa_pairs if q.get("_source") == "guidelines")
    druglabels_done  = sum(1 for q in qa_pairs if q.get("_source") == "druglabels")

    guidelines_needed  = max(0, GUIDELINES_N  - guidelines_done)
    druglabels_needed  = max(0, DRUGLABELS_N  - druglabels_done)

    logger.info(f"Need {guidelines_needed} more guidelines questions, {druglabels_needed} more druglabels questions.")

    # Load and filter KB chunks
    logger.info("Loading KB chunks...")
    g_records  = filter_chunks(load_jsonl(GUIDELINES_FILE) + load_jsonl(CDC_TB_FILE))
    d_records  = filter_chunks(load_jsonl(DRUGLABELS_FILE))
    logger.info(f"Usable chunks — guidelines: {len(g_records)}, druglabels: {len(d_records)}")

    random.seed(42)
    g_sample = random.sample(g_records, min(guidelines_needed * 3, len(g_records)))
    d_sample = random.sample(d_records, min(druglabels_needed * 3, len(d_records)))

    def run_generation(chunks, needed, source_label):
        generated = []
        idx = 0
        while len(generated) < needed and idx < len(chunks):
            chunk = chunks[idx]
            idx += 1
            text = get_text(chunk)
            source_hint = chunk.get("source", chunk.get("guideline", chunk.get("label_id", source_label)))
            qa = generate_qa(text, str(source_hint))
            if qa:
                qa["_source"] = source_label
                generated.append(qa)
                total_so_far = already_done + len(qa_pairs) - already_done + len(generated)
                logger.info(f"[{source_label}] {len(generated)}/{needed} — Q: {qa['question'][:80]}...")
            time.sleep(SLEEP_BETWEEN)
        return generated

    # Generate guidelines questions
    if guidelines_needed > 0:
        logger.info(f"\n── Generating {guidelines_needed} guidelines QA pairs ──")
        new_g = run_generation(g_sample, guidelines_needed, "guidelines")
        qa_pairs.extend(new_g)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
        logger.info(f"Cache saved: {len(qa_pairs)} total pairs.")

    # Generate druglabels questions
    if druglabels_needed > 0:
        logger.info(f"\n── Generating {druglabels_needed} druglabels QA pairs ──")
        new_d = run_generation(d_sample, druglabels_needed, "druglabels")
        qa_pairs.extend(new_d)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
        logger.info(f"Cache saved: {len(qa_pairs)} total pairs.")

    # Strip internal _source field and save final output
    final = [{"question": q["question"], "ground_truth": q["ground_truth"]} for q in qa_pairs]
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    logger.info(f"\n✅ Done! {len(final)} QA pairs saved to {OUTPUT_FILE}")
    logger.info(f"   Guidelines: {sum(1 for q in qa_pairs if q.get('_source')=='guidelines')}")
    logger.info(f"   Druglabels: {sum(1 for q in qa_pairs if q.get('_source')=='druglabels')}")


if __name__ == "__main__":
    main()
