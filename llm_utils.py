"""
llm_utils.py

Local generation utilities for the RAG pipeline.

This repo uses a small instruction-tuned causal LM for controlled extraction.
The RAG pipeline expects:
- `tokenizer`
- `model`
- `clean_llm_answer()`
- `generate_answer_model_only()`

Notes:
- For better format adherence, try "Qwen/Qwen2.5-1.5B-Instruct" or larger.
- This module loads the model at import time. If you want lazy loading
  (load only when needed), we can refactor it.
"""

from __future__ import annotations

import os
import re
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------------
# Model selection
# -------------------------
MODEL_NAME: str = os.getenv("LOCAL_QA_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

# DEVICE_MAP="auto" is great for CUDA. For CPU-only runs, "auto" can still work,
# but if it causes issues in your environment, set LOCAL_QA_DEVICE_MAP="cpu".
DEVICE_MAP: str = os.getenv("LOCAL_QA_DEVICE_MAP", "auto")

print(f"[llm_utils] Loading model: {MODEL_NAME}")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Make sure pad_token exists (some causal LMs don't define it)
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token

# Use float16 if CUDA is available; otherwise float32 for CPU stability
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    device_map=DEVICE_MAP,
)

model.eval()
print("[llm_utils] Model loaded.")


def clean_llm_answer(text: str) -> str:
    """
    Clean common decoding artifacts and reduce prompt echo.

    This function is intentionally conservative:
    - normalizes newlines
    - removes *some* common prompt headers if echoed
    - collapses excessive blank lines
    """
    if not isinstance(text, str):
        return ""

    # Normalize newlines safely (IMPORTANT: use literal escape sequences)
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    # Remove common prompt echo (keep conservative so we don't delete valid content)
    # Example: "You are a ... assistant" sometimes appears at the start.
    text = re.sub(
        r"^\s*(You are .*?assistant\.?\s*)",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    # Collapse excessive blank lines (3+ -> 2)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text


def _build_inputs(prompt: str):
    """
    Build model inputs.
    Prefer chat template if tokenizer supports it (Qwen Instruct does).
    """
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a clinical decision support assistant. "
                    "Follow the output format exactly."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return tokenizer(rendered, return_tensors="pt").to(model.device)

    # Fallback: plain prompt
    return tokenizer(prompt, return_tensors="pt").to(model.device)


def _generate(prompt: str, max_new_tokens: int = 220) -> str:
    """
    Internal generation with safer defaults (deterministic, reduced echo).
    """
    if not isinstance(prompt, str) or not prompt.strip():
        return ""

    inputs = _build_inputs(prompt)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            repetition_penalty=1.10,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    raw = tokenizer.decode(out[0], skip_special_tokens=True)
    return clean_llm_answer(raw)


def generate_answer_model_only(query: str, max_new_tokens: int = 200) -> str:
    """
    Model-only fallback (not evidence-grounded).

    In this project we generally avoid this for clinical safety.
    The RAG pipeline can still call this, but the default pipeline may ABSTAIN instead.
    """
    if not isinstance(query, str) or not query.strip():
        return ""

    prompt = (
        "You are a careful clinical assistant.\n"
        "If you cannot answer confidently, respond with: ABSTAIN\n\n"
        "QUESTION:\n"
        f"{query.strip()}\n\n"
        "ANSWER:\n"
    )

    return _generate(prompt, max_new_tokens=max_new_tokens)
