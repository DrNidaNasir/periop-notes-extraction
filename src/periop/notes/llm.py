"""Local language-model inference for clinical information extraction.

The model runs entirely on the local machine (no text leaves the computer), which
is the condition under which hospital notes can be processed. A small open model
is used here so that the pipeline runs on a laptop CPU:

    Qwen/Qwen2.5-0.5B-Instruct  (Apache-2.0 licence, 0.5 billion parameters)

Any instruction-tuned model that ships a chat template can be swapped in by
changing ``MODEL_DIR``.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .snippets import ANES_CLASSES, classify_anes_phrase

MODEL_DIR = Path("models/qwen2.5-0.5b-instruct")

SYSTEM = "You extract facts from surgical operative notes. Answer with the requested value only, no explanation."

# The wording of the two prompts was chosen on 30 development notes that are NOT part of
# the 60 evaluation notes (see docs/METHODS.md). A model of this size is sensitive to
# wording: variants that listed the fallback answer last made it answer "not stated" too often.
PROMPTS = {
    "ebl": (
        "{context}\n\n"
        "From the text above, extract the estimated blood loss. Give only the value "
        "(for example 200 mL, or minimal). Write \"not stated\" if it is missing."
    ),
    "anes": (
        "Operative note excerpt:\n{context}\n\n"
        "Question: What type of anesthesia was used? Quote it from the excerpt "
        "(examples: \"general endotracheal\", \"spinal\", \"local with sedation\"). "
        "If the excerpt does not say, write \"not stated\".\nAnswer:"
    ),
}


class LocalLLM:
    """Thin wrapper around a local causal language model with greedy decoding."""

    def __init__(self, model_dir: str | Path = MODEL_DIR, threads: int = 4, max_input_tokens: int = 3000):
        torch.set_num_threads(threads)
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
        self.model.eval()
        self.max_input_tokens = max_input_tokens

    @torch.no_grad()
    def ask(self, task: str, context: str, max_new_tokens: int = 24) -> tuple[str, float, int]:
        """Return (raw answer, seconds, number of prompt tokens) for one context."""
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPTS[task].format(context=context.strip())},
        ]
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt", truncation=True, max_length=self.max_input_tokens)
        t0 = time.time()
        out = self.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        text = self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip().split("\n")[0].strip(), time.time() - t0, int(ids["input_ids"].shape[1])


# --- normalisation of raw model answers into the label space ----------------------------
_VOL = re.compile(r"(?i)(\d{1,2},\d{3}|\d+(?:\.\d+)?)\s*(ml|mls|cc|cm3|l|liters?|litres?)\b")
_UNIT_TO_ML = {"ml": 1, "mls": 1, "cc": 1, "cm3": 1, "l": 1000, "liter": 1000, "liters": 1000, "litre": 1000, "litres": 1000}


def normalise_ebl(raw: str) -> str:
    """Map a raw answer to a number of mL (as a string), 'minimal' or 'not stated'.

    Convention shared with the rule baseline and the reference labels: if any volume is
    written (including "less than 50 mL"), the label is that number.
    """
    low = raw.lower()
    m = _VOL.search(raw)
    if m:
        return str(int(round(float(m.group(1).replace(",", "")) * _UNIT_TO_ML[m.group(2).lower()])))
    if re.search(r"minimal|negligible|none|nil|scant|trace|insignificant", low):
        return "minimal"
    return "not stated"


def normalise_anes(raw: str) -> str:
    """Map a raw answer to one of ANES_CLASSES using the same keyword rules as the baseline."""
    if "not stated" in raw.lower():
        return "not stated"
    label = classify_anes_phrase(raw)
    return label if label in ANES_CLASSES else "not stated"


NORMALISE = {"ebl": normalise_ebl, "anes": normalise_anes}
