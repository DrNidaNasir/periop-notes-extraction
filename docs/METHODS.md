# Methods

This file gives the definitions behind the numbers in the README, in enough detail to
reproduce or to criticise them.

## Part 1. Extraction from operative notes

### Data
MTSamples, a public collection of about 5,000 de-identified sample transcriptions.
Notes are kept if they come from a surgical specialty and contain an operative header
(ANESTHESIA, ESTIMATED BLOOD LOSS, PROCEDURE PERFORMED and similar). Exact duplicates
are removed. This leaves 881 operative notes (`periop/notes/select.py`).

### Variables
| Variable | Output |
|---|---|
| Estimated blood loss | a number in mL, or `minimal`, or `not stated` |
| Type of anaesthesia | `general`, `regional`, `local_or_sedation`, `combined`, `not stated` |

Conventions, applied identically to the reference labels, the rules and the model output:

* If any volume is written, the label is that number, including "less than 50 mL" (50).
* Words such as minimal, negligible, scant, nil or none give `minimal`.
* `regional` covers spinal, epidural, caudal, plexus and intravenous regional (Bier) blocks.
  `local_or_sedation` covers local infiltration, topical drops, eye and digital blocks,
  sedation and monitored anaesthesia care. `combined` is general together with regional.
  Local infiltration added to a general anaesthetic stays `general`.
* The anaesthesia may be stated in a header or only in the narrative ("after
  administration of general anesthesia"). Both count.

### Three methods
1. **Rules only.** Regular expressions read the value directly
   (`rule_ebl`, `rule_anes` in `periop/notes/snippets.py`).
2. **LLM on snippets.** A regular expression finds the mentions of the variable. A window of
   80 characters before and 220 after each hit (at most three windows) is passed to the
   model. If nothing is retrieved, the model is not called and the answer is `not stated`.
3. **LLM on the full note.** The whole note is passed to the model (truncated at 3,000 tokens).

The model is Qwen2.5-0.5B-Instruct, run locally on a laptop CPU with greedy decoding and
at most 24 new tokens. Its free-text answer is mapped to the label space by the same
keyword rules for all methods (`periop/notes/llm.py`).

### Prompt selection
Prompt wording was chosen on 30 development notes drawn from the notes that are not in
the evaluation sample, by agreement with the rule baseline. The evaluation notes were not
used for this.

### Reference labels
60 notes were drawn at random (seed 7) from the 881. For each note every passage
mentioning blood, bleeding, anaesthesia, sedation, local anaesthetic agents, blocks or
intubation was read, and the two variables were labelled according to the conventions
above (`annotations/reference_labels.csv`, with a comment on the difficult cases).
The labels were prepared with AI assistance and reviewed by the author. They are not
clinician annotations.

### Metrics
Exact agreement with the reference label, with a 95% bootstrap interval over notes.
Retrieval recall is the share of notes that state a value in which the regular
expression retrieved at least one snippet. Time is wall-clock inference time.

### Limits
Sample notes, not hospital records. Sixty notes give wide intervals. One small model, one
prompt per variable. Dictated sample reports are more regular than real clinical notes,
which favours the rules.

## Part 2. Early prediction of intra-operative hypotension

### Data
VitalDB, an open dataset of 6,388 non-cardiac operations at Seoul National University
Hospital (Lee et al., Scientific Data 2022). Cases are eligible if the patient is an adult,
had general anaesthesia lasting at least two hours, and has an invasive arterial pressure
waveform (`SNUADC/ART`). The first 200 eligible cases by case number are used, without
any random draw (`periop/waveform/download.py`).

### Prediction task
At time t, with data up to t only, predict hypotension five minutes later.

| Item | Definition |
|---|---|
| Input | last 30 s of arterial waveform (100 Hz, stored at 50 Hz) and last 5 min of MAP (1 Hz) |
| MAP | mean pressure over the trailing 10 s |
| Lead time | 5 min |
| Outcome | 1 if MAP stays below 65 mmHg for the whole minute from t + 5 min to t + 6 min |
| Sampling | one sample every 30 s |
| Primary cohort | samples where MAP at time t is 65 mmHg or higher |

### Signal quality
Values outside 20 to 250 mmHg and jumps of more than 40 mmHg between two samples are set
to missing. A sample is dropped if its input window has any missing value, or a pulse
pressure outside 15 to 150 mmHg in any 2-second block, or if the MAP history or the
outcome minute has gaps (`periop/waveform/segments.py`).

### Why no grey zone is removed
Several published models exclude samples whose future MAP lies between 65 and 75 mmHg.
That selection makes the two classes easier to separate than they are at the bedside, and
most of the apparent performance can then be obtained from the current MAP alone
(Enevoldsen and Vistisen, Anesthesiology 2022). Here all valid samples are kept, a
model using only the current MAP is reported as the reference, and the effect of the
grey-zone selection is shown as a sensitivity analysis.

### Models
| Name | Input | Model |
|---|---|---|
| `map_only` | current MAP | logistic regression |
| `record_only` | patient record | gradient boosting |
| `wave_features` | 20 waveform and MAP-trend features | gradient boosting |
| `wave_cnn` | raw 30-s waveform | 1-D CNN, 4 convolution blocks |
| `fusion` | waveform features + patient record | gradient boosting |

Waveform features are listed in `periop/waveform/features.py`. The patient record gives
numeric values (age, body-mass index, ASA class, emergency, hypertension, diabetes,
pre-operative laboratory values), categories (department, type of operation, approach,
position) and free text (operation name and diagnosis, encoded by TF-IDF and reduced to
8 components). Encoders are fitted on training patients only
(`periop/fusion/clinical.py`).

### Protocol
* Patients are split 60 / 20 / 20 into training, validation and test sets with a fixed
  seed. All samples of one patient are in one set.
* The number of boosting rounds, the CNN epoch and the decision threshold (Youden index)
  are chosen on the validation patients. Test patients are used once, for the reported
  numbers.
* Class imbalance is handled by class weights. No resampling.
* Intervals are 95% percentile intervals from 1,000 bootstrap resamples of test patients.
  Differences between models are computed on the same resamples.

* Robustness: the patient split is redrawn with ten seeds and the three tabular models
  are refitted each time (`periop/fusion/repeat.py`). Seed 0 is the split of the main tables.

### Limits
One hospital. 200 patients, so 40 test patients and wide intervals. Class weights make the
predicted probabilities too high; they would need recalibration before any use. Samples
30 s apart are strongly correlated. Treatment given by the anaesthetist in response to a
falling pressure changes the outcome and is not modelled. No external validation.
