"""Patient-record features for the multimodal model.

Three kinds of information from the VitalDB case table are attached to every
waveform sample of a case:

  numeric      age, body-mass index, ASA class, emergency flag, hypertension, diabetes
               and pre-operative laboratory values
  categorical  surgical department, type of operation, approach, position
  free text    the operation name and diagnosis as typed by the clinical team,
               turned into a few numeric components (TF-IDF followed by SVD)

Everything that is learned from data (category codes, vocabulary, SVD) is fitted on
the training patients only and then applied to the validation and test patients.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

NUMERIC = ["age", "bmi", "asa", "emop", "preop_htn", "preop_dm", "preop_hb", "preop_plt",
           "preop_na", "preop_k", "preop_alb", "preop_cr", "preop_gluc"]
CATEGORICAL = ["department", "optype", "approach", "position"]
TEXT = ["opname", "dx"]
N_TEXT_COMPONENTS = 8


class ClinicalEncoder:
    """Fit on training cases, transform any cases. Output is one numeric row per case."""

    def fit(self, cases: pd.DataFrame) -> "ClinicalEncoder":
        self.categories_ = {c: sorted(cases[c].dropna().astype(str).unique()) for c in CATEGORICAL}
        text = self._text(cases)
        self.tfidf_ = TfidfVectorizer(ngram_range=(1, 2), min_df=2, lowercase=True).fit(text)
        n_comp = min(N_TEXT_COMPONENTS, len(self.tfidf_.vocabulary_) - 1)
        self.svd_ = TruncatedSVD(n_components=n_comp, random_state=0).fit(self.tfidf_.transform(text))
        return self

    @staticmethod
    def _text(cases: pd.DataFrame) -> pd.Series:
        return (cases["opname"].fillna("") + " . " + cases["dx"].fillna("")).str.lower()

    def transform(self, cases: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=cases["caseid"].to_numpy())
        for c in NUMERIC:
            out[c] = pd.to_numeric(cases[c], errors="coerce").to_numpy()
        out["male"] = (cases["sex"].astype(str).str.upper().str[0] == "M").astype(float).to_numpy()
        for c in CATEGORICAL:
            # integer code per known category; a category unseen in training becomes missing
            codes = pd.Categorical(cases[c].astype(str), categories=self.categories_[c]).codes.astype(float)
            codes[codes < 0] = np.nan
            out["cat_" + c] = codes
        comps = self.svd_.transform(self.tfidf_.transform(self._text(cases)))
        for k in range(comps.shape[1]):
            out[f"text_{k}"] = comps[:, k]
        return out

    @property
    def groups(self) -> dict[str, list[str]]:
        """Column names by modality, used for the ablation (which modality adds what)."""
        return {
            "record_numeric": NUMERIC + ["male"],
            "record_categorical": ["cat_" + c for c in CATEGORICAL],
            "record_text": [f"text_{k}" for k in range(self.svd_.n_components)],
        }
