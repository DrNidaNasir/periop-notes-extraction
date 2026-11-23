"""Select operative (perioperative) notes from the public MTSamples corpus.

MTSamples is a collection of about 5,000 de-identified sample transcriptions.
We keep the reports that describe an operation: they come from surgical
specialties and contain an operative header such as ANESTHESIA, PROCEDURE or
ESTIMATED BLOOD LOSS. Exact duplicates (the corpus files many reports under two
specialties) are removed.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

SURGICAL_SPECIALTIES = {
    "Surgery",
    "Obstetrics / Gynecology",
    "Orthopedic",
    "Urology",
    "Neurosurgery",
    "Cardiovascular / Pulmonary",
    "ENT - Otolaryngology",
    "Ophthalmology",
    "Gastroenterology",
}

# A note is treated as an operative report if it has at least one of these headers.
OPERATIVE_HEADER = re.compile(
    r"(?i)\b(?:anesthesia|estimated blood loss|procedure performed|operation performed|"
    r"operative procedure|preoperative diagnos[ie]s|postoperative diagnos[ie]s)\s*:"
)


def load_operative_notes(csv_path: str | Path) -> pd.DataFrame:
    """Return one row per unique operative note with columns note_id, specialty, name, text."""
    df = pd.read_csv(csv_path)
    df = df.rename(columns={df.columns[0]: "note_id", "medical_specialty": "specialty",
                            "sample_name": "name", "transcription": "text"})
    df["specialty"] = df["specialty"].str.strip()
    df = df.dropna(subset=["text"])
    df = df[df["specialty"].isin(SURGICAL_SPECIALTIES)]
    df = df[df["text"].str.contains(OPERATIVE_HEADER)]
    # MTSamples separates lines with commas after full stops (".,"); restore line breaks
    df["text"] = df["text"].str.replace(r"\.,\s*", ".\n", regex=True).str.replace(r":,\s*", ": ", regex=True)
    df = df.drop_duplicates(subset="text").sort_values("note_id")
    return df[["note_id", "specialty", "name", "text"]].reset_index(drop=True)


if __name__ == "__main__":
    notes = load_operative_notes("data/mtsamples.csv")
    print(len(notes), "operative notes")
    print(notes["specialty"].value_counts().to_string())
