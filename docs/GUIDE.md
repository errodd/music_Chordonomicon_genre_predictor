# Chordonomicon genre predictor: end-to-end guide

This guide has three parts. Part 1 explains the ideas the project depends on, using chord and harmony examples wherever possible. Part 2 maps the repository and follows the data from the CSV files to the final score. Part 3 is a step-by-step run sheet. If you only need to run the pipeline, jump to Part 3, but read section 1.6 first. It is the one idea the whole final check rests on.

---

# Part 1. The concepts

## 1.1 What the project does

The task is to predict a song's genre from its chords alone. The input is one string such as:

```
<intro_1> C <verse_1> F C E7 Amin C F C G7 C <chorus_1> F C F C G C
```

The output is one of 12 genres: alternative, country, electronic, jazz, metal, pop, pop rock, punk, rap, reggae, rock, soul.

This is **classification**: the model chooses one label from a fixed list. Training means showing the model many examples where the answer is known, so it can find patterns that connect chords to genres. Then you give it a new string and it guesses.

**Notation used in the data.** The dataset has its own spelling, and the code depends on it:

| Written in data | Means |
|---|---|
| `Amin`, `Cmin7` | A minor, C minor seventh (the data never writes `Am`) |
| `Fsmin` | F♯ minor (`s` means sharp, `b` means flat) |
| `Gmaj7`, `E7`, `Gsus4`, `Bdim` | what you expect |
| `E5` | power chord (root and fifth) |
| `A/Cs` | A over a C♯ bass (slash chord) |
| `<verse_1>`, `<chorus_2>` | section markers, not chords |

**The label is not really about the song.** Each track carries the genre that Spotify gives its *artist*, copied to every track by that artist. A rock-sounding song by a country artist is labeled `country`. This is a proxy label. It limits how good any model can get, and it is also why the split must be done carefully (section 1.6).

## 1.2 Features: turning chords into numbers

A model can only work with numbers, so the chord string must become a list of numbers. Those numbers are called **features**. The project builds two families of features. They are compared against each other and are never mixed.

### The aggregate features (13 numbers per song)

These are summary statistics, like describing a piece by its length and texture:

- how many chords (`n_chords`), how many distinct chords, their ratio
- the share of chords that are major, minor, power, suspended, diminished, augmented, or unreadable (`share_major`, `share_minor`, ...)
- `abab_share`: how often a 4-chord window alternates between two chords, like `C G C G`
- `has_structure`: whether section markers like `<verse_1>` appear

Traced example. In `E E E E A A E E B A E B`, there are 12 chords and 3 distinct ones, so the ratio is 3 / 12 = 0.25. All 12 chords are major, so `share_major` = 1.0. In a ballad that uses `Amin`, `Emin`, `F`, `C`, `share_minor` would be well above zero. In a 20,000-song sample of the real data, about 72% of chords were read as major and 25% as minor.

### The n-gram features (up to 5,000 numbers per song)

An **n-gram** is a run of n chords in a row. This is the same idea as looking at fragments of a progression: a two-chord n-gram is a pair, a four-chord one is a short progression. For each song the model counts how often each fragment appears.

```
chords:   C  F  C  G  C  F
2-grams:  (C F) (F C) (C G) (G C) (C F)      -> "C F" appears twice
```

The vocabulary (which fragments count as features) is learned from training songs: keep fragments that appear in at least 5 songs, and keep the 5,000 most common. The project tests orders 1 to 4 in two **alphabets**.

**Absolute alphabet** uses the chord names as written, so `D C G` and `G F C` are different fragments, even though they are the same progression in different keys.

**Degree alphabet** replaces each chord with its role in the key, so transposed songs look the same. This is the Roman numeral idea, and it needs the song's key.

### How the code finds the key (traced)

The code guesses the key with the Krumhansl–Kessler profiles, a standard way to measure how well a set of pitch classes fits a major or minor key. Steps:

1. Count how often each root (C, C♯, ... B) appears in the song.
2. Compare that histogram to the profile of each of the 24 keys, using correlation.
3. Pick the key with the highest correlation, then label every chord by its distance above that tonic.

Traced run on `D C G D D C G D`:

- Root histogram (C to B): C=2, D=4, G=2, everything else 0.
- Best match: G major, correlation 0.729. Next best: D major, 0.545.
- Chord roles in G: D is degree 5, C is degree 4, G is degree 1. Result: `5:major 4:major 1:major 5:major ...`

So this song is analyzed as V–IV–I in G, not I–♭VII–IV in D. Both readings describe the same chords. What matters for the model is only that `D C G` and `G F C` give identical degree fragments, which is the point of the degree alphabet.

**A known weakness.** The key finder looks at a histogram, so it ignores the order of chords. Traced on `C Amin F G C Amin F G` (a I–vi–IV–V that you would call C major):

- Root histogram: C=2, F=2, G=2, A=2.
- Best match: F major, 0.766. C major is a close second at 0.750.
- Resulting degrees: `5 3m 1 2` (read in F) instead of `1 6m 4 5` (read in C).

Short loops with equal chord counts are ambiguous for this method. This is a stated limitation, not a bug.

## 1.3 The models

A model is the rule that maps features to a genre. This project compares four. All are simple and fast, and all work on a table of numbers.

- **DummyClassifier** always answers the most common genre (pop). It is not a real model. It sets the floor: anything useful must beat it.
- **ComplementNB** (naive Bayes) estimates, for each genre, how often each chord fragment appears, then picks the genre whose habits best fit the song. It assumes fragments are independent, which is not true of music, but it is quick and robust. It needs non-negative counts, so it only works on n-grams.
- **SGDClassifier** (logistic regression trained by small steps) learns one weight per feature per genre. A positive weight means "this fragment points toward that genre". It is the one that gets tuned.
- **LinearSVC** (a linear support vector machine) draws a separating boundary for each genre against the rest. It is the slowest here. It is capped at 100 iterations because at 1,000 it did not converge and ran for hours.

All four are **linear**: each feature pushes the answer up or down independently. They cannot represent interactions such as "this fragment means something only after that one".

## 1.4 Scores: what the numbers mean

**Accuracy** is the fraction of songs guessed right. It misleads here because pop is about 24% of the data. A model that always says "pop" scores about 24% without learning anything.

**Macro-F1** is the project's main score. For each genre, F1 combines two questions: of the songs the model called this genre, how many were right (precision), and of the songs that really are this genre, how many did it find (recall). Macro-F1 averages the 12 F1 values with equal weight, so getting reggae wrong costs as much as getting pop wrong.

Traced for the dummy: it says pop every time. For pop, precision is 0.24 (24% of songs really are pop) and recall is 1.0, so F1 is about 0.39. For each of the other 11 genres F1 is 0. The average is 0.39 / 12 = 0.0325, which matches what the notebook measured for the dummy (0.0325).

**Chance level** for 12 classes is about 1 / 12 = 0.083 for accuracy. An earlier run of the best model reached macro-F1 of about 0.14 and accuracy of about 0.27. That is clearly above the floor and still weak. Part of the reason is the proxy label, and part is that chord progressions alone do not separate genres sharply. Your results after the rerun will differ somewhat, because the data split changed.

## 1.5 Hyperparameters and Optuna

A **hyperparameter** is a setting you choose before training, such as how strongly to penalize large weights (`alpha`), which penalty type to use, and whether to give rare genres extra weight (`class_weight`). It is like choosing an EQ curve before mixing. The learned weights are the result of training, and the hyperparameters are decided outside it.

**Optuna** automates the search. It runs 20 trials. Each trial picks a setting, trains on the training part of each fold, and scores on the validation part. Optuna uses past trials to choose promising settings next (the TPE method). The best setting is then compared against the default one. If the improvement is smaller than the natural fold-to-fold variation, the project keeps the simpler default.

## 1.6 Leakage, splits, and holdouts (the central idea)

This section explains the problem the corrections are about.

### The goal

You want to know how the model will do on music it has never seen. An honest estimate needs a test where the model truly has no prior information about the answers.

### Leakage

**Leakage** means information from the test side gets into training or into your decisions, so the score is better than reality. It is like a student who sees the exam questions before the exam and then reports the grade as a measure of learning.

Leakage here can happen in several ways:

1. **Same artist on both sides.** If an artist has 40 tracks all labeled `country`, and 32 land in training and 8 in test, the model can learn that artist's style and look good on the 8 test tracks without learning anything about country music. The fix is to split by artist, so all of an artist's tracks go to one side. This was done from the start with `GroupShuffleSplit`.
2. **Looking at test labels.** If you count or inspect the test labels, you may unconsciously shape your choices to them.
3. **Reusing the test set.** If you evaluate on the test, change something, and evaluate again, then the test has become part of your design process. It is no longer a test.
4. **Exploring the test data before deciding features.** If the EDA (exploratory analysis) looked at all rows, including those that later became the test, then feature ideas came partly from the test.
5. **Same song under different artists.** If the exact chord string appears for two artists, one in training and one in test, the model can recognize a song it has seen.
6. **Fitting preprocessing on everything.** For example, learning the n-gram vocabulary from all songs including the validation part. Inside cross-validation, the vocabulary must be learned only from that fold's training part.

### The data sets in this project

```
all 679,807 tracks
   ├── 352,111 with a genre label  (38,057 artists)
   │      ├── HOLDOUT   72,478 tracks / 7,612 artists   (final exam, used once)
   │      └── DEV      ~279,618 tracks / 30,443 artists (everything else)
   └── 327,696 without a genre label  (used only in the EDA, never for training)
```

- **Dev** (development) data is everything you may use to build the model. Inside it, **cross-validation** does the choosing.
- **Holdout** is set aside once, before further decisions, and used for a single final score.

### Cross-validation

The dev data is cut into 3 folds, always by artist. For each fold: train on the other two, score on this one. The three scores are averaged and their spread (standard deviation) is kept. Use this to compare representations, models and hyperparameters, as often as you like. This is called validation, and the folds play the role of a rehearsal.

The rule for picking a winner: take the best mean score. If the gap to the runner-up is smaller than the winner's own spread across folds, the difference is noise, so keep the simpler option.

### What was wrong, and what we changed

The reviewer's corrections described these problems. Here is how each was handled.

| Problem | What we did | Status |
|---|---|---|
| The old test set had been evaluated, and its labels were printed | The old test became dev data. A new holdout was drawn with a new seed (2026), frozen in `data/holdout_artists.json`, with a hash in `data/holdout_manifest.json` | Fixed |
| The notebooks could read the holdout labels at any time | After the split, only row positions are kept. The labels are loaded in the final cell, after the model is fitted | Fixed |
| The EDA had seen every row | The EDA now removes the holdout (and its exact duplicates) before exploring anything | Fixed going forward. The *earlier* EDA run did see those rows, which cannot be undone, so it is stated as a limitation |
| The same chord string under different artists | Dev rows whose exact `chords` string also appears in the holdout are dropped (315 rows, 15 of them labeled) | Fixed for exact matches only. Transposed or reordered copies are not detected |
| Earlier decisions used the old split | Notebook 03 reruns all choices on the new dev data | Redone, but earlier results were seen once. Stated as a limitation |
| Code copied between notebooks | Shared code moved into `src/chordmodel.py` and `src/chorddata.py` | Done for notebook 03. Notebook 02 still has its own copy of the transformers |

### Why the holdout is used once

If you look at the holdout score, then change the model, then look again, you have turned the holdout into another validation set. The score would then be optimistic. The final cell is written to run once per kernel. Treat the number it prints as the final answer, even if you do not like it.

## 1.7 Pipelines and why there is a module

A **pipeline** chains the steps (read chords, build n-grams, run the classifier) into one object. It makes sure that anything learned from data, like the vocabulary, is learned only inside the fold being trained. Then `pipeline.predict(new_songs)` does everything in one call.

The pipeline classes used to live inside the notebook. A saved model then could not be reloaded outside that notebook, because Python remembers where each class was defined. Moving the classes to `src/chordmodel.py` fixes this: with `src/` on the Python path, `joblib.load("model.joblib")` works anywhere.

---

# Part 2. The repository, and how data flows

## 2.1 Files

| Path | What it is |
|---|---|
| `data/chordonomicon_v2_part_0..6.csv` | The dataset, split into 7 pieces (the big `chordonomicon_v2.csv` is git-ignored) |
| `data/make_holdout.py` | Draws the holdout once and writes the two JSON files. Refuses to run if they exist |
| `data/holdout_artists.json` | The 7,612 holdout artist ids. **Do not open to look at labels, and do not edit** |
| `data/holdout_manifest.json` | Seed, sizes, duplicate counts, hash of the artist list, package versions |
| `src/chorddata.py` | Loads the CSVs and decides which rows are dev and which are holdout |
| `src/chordmodel.py` | Chord parsing, both transformers, model and pipeline factories |
| `notebooks/EDA.ipynb` | Exploratory analysis on dev data only |
| `notebooks/02_data_preparation.ipynb` | Defines and checks the feature transformers, closes the preparation decisions |
| `notebooks/03_modeling_and_model_selection.ipynb` | Comparison, tuning, selection, the single holdout evaluation, error analysis |
| `requirements.txt` | Python packages |
| `.venv/` | The virtual environment (ignored by git) |
| `model.joblib` | The saved final pipeline (created by notebook 03) |

## 2.2 The flow

1. **`make_holdout.py`** (run once, already done): reads the CSVs, draws 20% of labeled artists as the holdout, removes exact duplicate chord strings from dev, and records everything in the manifest with a hash.
2. **EDA**: loads all rows, immediately drops the holdout and duplicates, then explores dev data. Its findings motivate the features.
3. **Notebook 02**: defines the feature transformers, loads the same split, and checks that the pipelines behave (no excluded columns, no NaN, no leakage of the target).
4. **Notebook 03**:
   1. Loads and validates the CSVs, builds the dev set, keeps only the holdout row positions, and deletes everything that holds holdout labels.
   2. Builds 3 grouped folds on dev.
   3. Screens the 8 n-gram variants with ComplementNB (orders 1 to 4, absolute and degree) and picks one.
   4. Compares the four models on that winner, plus a control on the 13 aggregate features.
   5. Tunes SGD with Optuna.
   6. Applies the selection rule and chooses a final pipeline.
   7. Fits it on all dev data, then loads the holdout labels and scores once.
   8. Reads the errors: per-genre F1 and the confusion matrix.
   9. Saves the model to `model.joblib`.

---

# Part 3. How to run the whole pipeline

## 3.1 One-time setup

The environment `.venv` and the kernel `Python (chordonomicon)` already exist on this machine. To confirm, from the repo root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -c "import pandas, numpy, scipy, sklearn, optuna, matplotlib; print('ok')"
```

It must print `ok`. If it does not, install the packages:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt scipy
```

On a new machine, first create the environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt scipy
.\.venv\Scripts\python.exe -m ipykernel install --user --name chordonomicon --display-name "Python (chordonomicon)"
```

The holdout files `data/holdout_artists.json` and `data/holdout_manifest.json` must exist. They are already created. **Do not rerun `make_holdout.py`.** It refuses to run when they exist, and you should not delete them.

## 3.2 Choose the kernel (the step that went wrong earlier)

In VS Code, open each notebook and click the kernel name at the top right. Choose `.venv (3.10.6)` (`.venv\Scripts\python.exe`) or `Python (chordonomicon)`. Do **not** choose `Python 3.13.7`. It has no Optuna and the imports will fail.

## 3.3 Run order

Run each notebook with **Restart** then **Run All** (or the "Restart and Run All" command), from a fresh kernel. The corrections ask for consecutive execution counts and saved outputs, which only a clean full run gives.

1. **`EDA.ipynb`**. About 15 minutes (a headless run measured 888 seconds, with no errors). The notebook already holds the outputs of that clean run, with execution counts 1 to 31 and its written figures updated to the dev numbers, so rerunning it is optional. If you rerun it, look for the new cell "Corte de desarrollo", which prints the full and dev row counts. The dev count must be 607,014.
2. **`02_data_preparation.ipynb`**. Takes roughly 20 to 40 minutes, because it fits the n-gram vocabularies on the full training data (the notebook's own notes measured about 22 minutes for eight of them). Look for "Filas de dev descartadas por repetir un chords del holdout: 15" and train of 279,618 rows.
3. **`03_modeling_and_model_selection.ipynb`**. This is the long one, about 1.5 to 3 hours. Leave the computer on, plugged in, and do not run other heavy programs. Watch for these lines as it progresses:
   - `Holdout: 72,478 filas | 7,612 artistas (sin etiquetas en memoria)`
   - one line per n-gram variant in the screen, with its macro-F1 and cache time
   - the model comparison table
   - the Optuna result and default vs tuned comparison
   - `Pipeline final: ... Test: macro-F1 ...` (the single holdout evaluation)
   - the per-genre table and the confusion matrix

## 3.4 Rules while running

- **Run notebook 03 top to bottom without skipping.** The error-analysis cell stops with an error if the final cell has not run.
- **Run the final evaluation cell once.** If you restart the kernel and run it again, you are re-using the holdout. Write down the first number you get.
- **Do not change the model, the features, or the settings after you see the holdout score.** If something is broken (crash, wrong sizes), that is a bug and you may fix it. If you just dislike the number, that is not a reason to change anything.
- **Do not open `holdout_artists.json` to look up artists, and do not look at holdout labels.**
- If the kernel freezes for a long time in the LinearSVC cell or in the n-gram cache, check the RAM in Task Manager. Close other applications. If memory is full, interrupt the kernel (not restart), then rerun that cell.

## 3.5 After the run

1. Confirm `model.joblib` was written at the repo root.
2. Try a prediction in a new notebook cell or in a script (the model works anywhere once `src/` is on the path):

```python
import sys, joblib, pandas as pd
sys.path.insert(0, "src")           # if running from the repo root; use "../src" from notebooks/
model = joblib.load("model.joblib")

new = pd.DataFrame({"chords": [
    "<verse_1> E E E E A A E E B A E B",
]})
print(model.predict(new))
```

3. Commit the new files: `src/`, `data/make_holdout.py`, the two holdout JSON files, `docs/GUIDE.md`, and the rerun notebooks.

## 3.6 What to write in the final report

Include these points. They are the limits that cannot be removed with code:

1. The first EDA run looked at all rows, including the holdout. The delivered EDA and everything after it uses dev data only.
2. Early modeling decisions were first made on the old split, and those results were seen once. They were redone on the new dev data, but the earlier numbers were not unseen.
3. Only exact duplicate chord strings were removed. Transposed or reordered copies of the same song may remain.
4. The label is the artist's Spotify genre, copied to each track, so some labels are wrong for individual songs.
5. All four models are linear. Interactions between chord fragments are not modeled.
6. The key is estimated from a root histogram and can be ambiguous for short loops.
7. Three folds give a wide spread, so small differences between candidates are not real differences.

---

# Glossary

- **Artist-grouped split**: a split where all of one artist's tracks go to the same side.
- **Baseline**: a trivial reference model that any real model must beat.
- **Class**: one of the labels the model chooses from (here, a genre).
- **Confusion matrix**: a table of true genre against predicted genre. The diagonal is correct answers.
- **Cross-validation (CV)**: repeated train/validate cycles over folds of the dev data.
- **Dev data**: all data you are allowed to use while building the model.
- **Feature**: a number the model receives as input.
- **Fold**: one part of the dev data held back for validation in one CV round.
- **Holdout**: data set aside and used once, at the very end.
- **Hyperparameter**: a setting chosen before training.
- **Leakage**: information from the test side reaching training or decisions.
- **Macro-F1**: the average of F1 over classes, weighting each class equally.
- **N-gram**: a run of n consecutive chords, used as a feature.
- **Pipeline**: the chained steps from raw input to prediction.
- **Precision / recall**: of the songs called X, the fraction right; of the real X songs, the fraction found.
- **Proxy label**: a label standing in for the real target (here, artist genre standing for song genre).
- **Vocabulary**: the list of n-grams a transformer counts. It is learned from training data only.
