"""Regular-expression snippet retrieval and rule-based baselines.

Idea (after SPELL, Kleinlein et al., Comput Methods Programs Biomed 2026):
a long clinical note contains only one or two sentences that matter for a given
variable. A cheap regular expression finds those sentences; only that short
"snippet" is then passed to the language model. This file holds

  * the retrieval patterns for two perioperative variables, and
  * a rule-only extractor for each variable, used as the baseline that the
    language model has to beat.

Variables
---------
ebl   estimated blood loss. Output: a number in mL, or "minimal", or "not stated".
anes  type of anaesthesia.  Output: one of ANES_CLASSES.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

ANES_CLASSES = ["general", "regional", "local_or_sedation", "combined", "not stated"]

# --- retrieval patterns -----------------------------------------------------------------
RETRIEVAL = {
    "ebl": re.compile(r"(?i)\b(estimated blood loss|blood loss|EBL)\b"),
    "anes": re.compile(r"(?i)\b(an[a]?esthesia|an[a]?esthetic)\b"),
}

WINDOW_BEFORE = 80   # characters kept before the match
WINDOW_AFTER = 220   # characters kept after the match


@dataclass
class Snippet:
    start: int
    end: int
    text: str


def retrieve(text: str, task: str, max_snippets: int = 3) -> list[Snippet]:
    """Return up to ``max_snippets`` non-overlapping context windows around pattern hits."""
    out: list[Snippet] = []
    for m in RETRIEVAL[task].finditer(text):
        s, e = max(0, m.start() - WINDOW_BEFORE), min(len(text), m.end() + WINDOW_AFTER)
        if out and s <= out[-1].end:           # merge overlapping windows
            out[-1] = Snippet(out[-1].start, e, text[out[-1].start:e])
        else:
            out.append(Snippet(s, e, text[s:e]))
        if len(out) >= max_snippets:
            break
    return out


# --- rule-only baselines ----------------------------------------------------------------
_NUM = r"(\d{1,2},\d{3}|\d+(?:\.\d+)?)"
_EBL_VALUE = re.compile(
    r"(?i)(?:estimated blood loss|blood loss|EBL)[^.\n]{0,60}?"
    r"(?:(less than|<|approximately|about|around|was|:)\s*)*" + _NUM + r"\s*(ml|mls|cc|cm3|l|liters?|litres?)\b"
)
_EBL_MINIMAL = re.compile(r"(?i)(?:estimated blood loss|blood loss|EBL)[^.\n]{0,40}?(minimal|negligible|nil|none|scant|trace|insignificant)")
_UNIT_TO_ML = {"ml": 1, "mls": 1, "cc": 1, "cm3": 1, "l": 1000, "liter": 1000, "liters": 1000, "litre": 1000, "litres": 1000}


def rule_ebl(text: str) -> str:
    """Rule-only estimated blood loss: first numeric value with a volume unit after the key phrase."""
    m = _EBL_VALUE.search(text)
    if m:
        value = float(m.group(2).replace(",", "")) * _UNIT_TO_ML[m.group(3).lower()]
        return str(int(round(value)))
    if _EBL_MINIMAL.search(text):
        return "minimal"
    return "not stated"


_ANES_FIELD = re.compile(r"(?i)\ban[a]?esthesia\s*:\s*([^\n]{0,160})")
_GENERAL = re.compile(r"(?i)\b(general|endotracheal|GETA|LMA|laryngeal mask)\b")
_REGIONAL = re.compile(r"(?i)\b(spinal|epidural|regional|nerve block|interscalene|axillary block|femoral block|bier block|subarachnoid|caudal)\b")
_LOCAL = re.compile(r"(?i)\b(local|MAC|monitored an[a]?esthesia care|sedation|topical|lidocaine|marcaine|xylocaine|retrobulbar|peribulbar)\b")


def classify_anes_phrase(phrase: str) -> str:
    """Map a free-text anaesthesia phrase to one of ANES_CLASSES with keyword rules."""
    g, r, l = bool(_GENERAL.search(phrase)), bool(_REGIONAL.search(phrase)), bool(_LOCAL.search(phrase))
    if g and r:
        return "combined"
    if g:
        return "general"
    if r:
        return "regional"
    if l:
        return "local_or_sedation"
    return "not stated"


def rule_anes(text: str) -> str:
    """Rule-only anaesthesia type: classify what follows the first 'ANESTHESIA:' header."""
    m = _ANES_FIELD.search(text)
    return classify_anes_phrase(m.group(1)) if m else "not stated"


RULES = {"ebl": rule_ebl, "anes": rule_anes}
