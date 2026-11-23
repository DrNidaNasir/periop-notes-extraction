"""Compare three ways of extracting a variable from operative notes.

    rules        regular expressions only (no language model)
    llm_snippet  regular expressions retrieve short snippets; the local LLM reads only those
    llm_full     the local LLM reads the whole note

Each method is scored against hand-checked reference labels
(``annotations/reference_labels.csv``) on exact agreement, and timed.

Usage:
    python -m periop.notes.evaluate --out results/notes
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .llm import NORMALISE, LocalLLM
from .select import load_operative_notes
from .snippets import RULES, retrieve

TASKS = ["ebl", "anes"]


def run(notes_csv: Path, labels_csv: Path, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    notes = load_operative_notes(notes_csv).set_index("note_id")
    labels = pd.read_csv(labels_csv, dtype=str).fillna("")
    labels["note_id"] = labels["note_id"].astype(int)
    llm = LocalLLM()

    rows = []
    for k, ref in enumerate(labels.itertuples(index=False), 1):
        text = notes.loc[ref.note_id, "text"]
        for task in TASKS:
            truth = getattr(ref, task)

            t0 = time.time()
            pred_rules = RULES[task](text)
            sec_rules = time.time() - t0

            snips = retrieve(text, task)
            if snips:
                raw_s, sec_s, tok_s = llm.ask(task, "\n...\n".join(s.text for s in snips))
                pred_snip = NORMALISE[task](raw_s)
            else:                       # nothing retrieved: the model is not called at all
                raw_s, sec_s, tok_s, pred_snip = "", 0.0, 0, "not stated"

            raw_f, sec_f, tok_f = llm.ask(task, text)
            pred_full = NORMALISE[task](raw_f)

            rows.append(dict(note_id=ref.note_id, task=task, truth=truth, n_chars=len(text), retrieved=bool(snips),
                             rules=pred_rules, sec_rules=sec_rules,
                             llm_snippet=pred_snip, raw_snippet=raw_s, sec_snippet=sec_s, tok_snippet=tok_s,
                             llm_full=pred_full, raw_full=raw_f, sec_full=sec_f, tok_full=tok_f))
        print(f"[{k}/{len(labels)}] note {ref.note_id} done", flush=True)
        pd.DataFrame(rows).to_csv(out_dir / "predictions.csv", index=False)   # save as we go
    return pd.DataFrame(rows)


def bootstrap_ci(correct: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval for an accuracy (resampling notes)."""
    rng = np.random.default_rng(seed)
    means = [correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise(pred: pd.DataFrame, out_dir: Path) -> dict:
    summary: dict = {}
    for task, g in pred.groupby("task"):
        stated = g["truth"] != "not stated"
        entry = {"n_notes": int(len(g)), "n_with_value": int(stated.sum()),
                 # of the notes that do state a value, how many did the regular expression retrieve?
                 "retrieval_recall": float(g.loc[stated, "retrieved"].mean())}
        for method, sec in [("rules", "sec_rules"), ("llm_snippet", "sec_snippet"), ("llm_full", "sec_full")]:
            correct = (g[method] == g["truth"]).to_numpy()
            lo, hi = bootstrap_ci(correct)
            entry[method] = {"accuracy": float(correct.mean()), "ci95": [lo, hi],
                             "accuracy_when_value_stated": float(correct[stated.to_numpy()].mean()),
                             "seconds_total": float(g[sec].sum()), "seconds_per_note": float(g[sec].mean())}
        entry["time_saved_snippet_vs_full"] = 1 - entry["llm_snippet"]["seconds_total"] / entry["llm_full"]["seconds_total"]
        summary[task] = entry
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", type=Path, default=Path("data/mtsamples.csv"))
    ap.add_argument("--labels", type=Path, default=Path("annotations/reference_labels.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/notes"))
    ap.add_argument("--summary-only", action="store_true", help="recompute the summary from saved predictions")
    a = ap.parse_args()
    preds = pd.read_csv(a.out / "predictions.csv", dtype={"truth": str, "rules": str, "llm_snippet": str, "llm_full": str}) \
        if a.summary_only else run(a.notes, a.labels, a.out)
    print(json.dumps(summarise(preds, a.out), indent=2))
