# Chord representation, key estimation and Roman numerals

This document explains how [`notebooks/EDA.ipynb`](../notebooks/EDA.ipynb) turns the raw `chords` strings of Chordonomicon into three representations:

1. **Absolute chords**: the chord symbols as written (`C G Amin F`), cleaned and classified by quality.
2. **Roman numerals**: each chord relative to the song's estimated key (`I V vi IV`).
3. **Intervals**: each chord relative to the previous one (`+7maj +2min +8maj`).

It follows one song from the CSV to the per-genre counts, with the code of each step.

**Code citations** have the form *Section N · cell C · L a–b*: the notebook section, the cell's position in the notebook (0-based, as stored in the `.ipynb` file), and the line numbers inside that cell. Code blocks are copied verbatim from the notebook.

---

## Contents

1. [Pipeline at a glance](#1-pipeline-at-a-glance)
2. [The raw data](#2-the-raw-data)
3. [Choosing songs and chord tokens](#3-choosing-songs-and-chord-tokens)
4. [Parsing a chord symbol](#4-parsing-a-chord-symbol)
5. [Chord quality](#5-chord-quality)
6. [From text to numbers](#6-from-text-to-numbers)
7. [Key estimation](#7-key-estimation)
8. [Roman numerals](#8-roman-numerals)
9. [Intervals: the key-free representation](#9-intervals-the-key-free-representation)
10. [Four-chord windows and per-genre counts](#10-four-chord-windows-and-per-genre-counts)
11. [What the representations show on the corpus](#11-what-the-representations-show-on-the-corpus)
12. [Known limitations and possible improvements](#12-known-limitations-and-possible-improvements)
13. [Quick reference](#13-quick-reference)

---

## 1. Pipeline at a glance

```
CSV row: "<verse_1> C G Amin F <chorus_1> F C/E Gs/ G"
   │
   │  split_tokens            remove part tags            (Section 8)
   ▼
["C","G","Amin","F","F","C/E","Gs/","G"]
   │
   │  parse_chord +           root / suffix / bass,       (Section 9)
   │  chord_quality           suffix → quality
   │
   │  chord_pc                root → pitch class 0–11,    (Section 10)
   ▼                          drop unknown tokens
[(0,"major"), (7,"major"), (9,"minor"), (5,"major"), (5,"major"), (0,"major"), (7,"major")]
   │
   │  estimate_key            try 12 major keys, keep the best
   ▼
key = 0 (C major), key fit = 1.0
   │
   ├── to_numeral    →  I  V  vi  IV  IV  I  V
   └── to_intervals  →  +7maj +2min +8maj +0maj +7maj +7maj
   │
   │  4-chord windows, ABAB skipped
   ▼
quads_rn_by_genre["pop"][("I","V","vi","IV")] += 1  …
```

This song is used as the running example throughout. It was chosen because it touches every branch of the code: two part tags, a slash chord (`C/E`) and a malformed token (`Gs/`).

---

## 2. The raw data

Each row of Chordonomicon has a `chords` column: **one string per song**, holding the whole song in order. It mixes two kinds of tokens separated by spaces:

- **Part tags** in angle brackets: `<intro_1>`, `<verse_2>`, `<chorus_1>`, … Eight families in total (intro, verse, chorus, bridge, interlude, solo, instrumental, outro).
- **Chord symbols** in the corpus's own spelling:

| Spelling | Meaning |
|---|---|
| `C`, `G`, `Bb` | major triads (`b` = flat) |
| `Cs`, `Fs` | C♯, F♯ (**`s` = sharp**, not `#`) |
| `Amin`, `Fsmin` | minor triads |
| `E7`, `Cmaj7`, `Bmin7` | seventh chords |
| `Dsus4`, `Asus2` | suspended |
| `Eno3d` | power chord (E5: root and fifth, **no third**) |
| `Bdim`, `Bdim7`, `Bdimb7` | diminished, fully diminished 7th, half-diminished (m7♭5) |
| `A/Cs` | slash chord: A major over C♯ in the bass |

The whole corpus uses **4,314 distinct spellings**, built from 17 roots and 56 suffixes.

---

## 3. Choosing songs and chord tokens

### 3.1 Which songs

Only songs that have a genre label are used for the genre analysis. Two parallel arrays are built once:

*Section 9 · cell 44 · L4–5*
```python
labeled_chords = df.loc[df["main_genre"].notna(), "chords"].to_numpy()
labeled_genres = df.loc[df["main_genre"].notna(), "main_genre"].to_numpy()
```

- `df["main_genre"].notna()` gives True/False for each of the 679,807 rows.
- `df.loc[mask, "chords"]` keeps the True rows and only the `chords` column.
- `.to_numpy()` turns the result into a plain array, which is faster to loop over.

The result is **352,111 songs**. Song *i*'s chord string is `labeled_chords[i]` and its genre is `labeled_genres[i]`.

### 3.2 Which tokens: removing part tags

*Section 8 · cell 33 · L1, L5–16*
```python
PART_RE = re.compile(r"^<([A-Za-z]+)_(\d+)>$")


def split_tokens(chords: str) -> tuple[list[str], list[str]]:
    # Return (chord_tokens, part_names_lower) for one track.
    chord_toks: list[str] = []
    parts: list[str] = []
    for tok in str(chords).split():
        if tok.startswith("<") and tok.endswith(">"):
            m = PART_RE.match(tok)
            # Nonstandard tags such as <intro_riff_1> keep their name so they show up in unknown_parts.
            parts.append(m.group(1).lower() if m else re.sub(r"_\d+$", "", tok[1:-1]).lower())
        else:
            chord_toks.append(tok)
    return chord_toks, parts
```

| Line | What it does | Running example |
|---|---|---|
| `str(chords).split()` | cuts the string at every space | `["<verse_1>", "C", "G", "Amin", "F", "<chorus_1>", "F", "C/E", "Gs/", "G"]` |
| `if tok.startswith("<") and tok.endswith(">")` | is this token a part tag? | true for `<verse_1>` and `<chorus_1>` |
| `parts.append(...)` | stores the tag name without its number (`verse_1` → `verse`) | `parts = ["verse", "chorus"]` |
| `chord_toks.append(tok)` | everything else is a chord | `["C", "G", "Amin", "F", "F", "C/E", "Gs/", "G"]` |

**What is passed on is the whole song**: every section, in order, with every repetition. There is no selection of "important" chords. A chord played 20 times appears 20 times, and so weighs 20 times in the key estimate. Chord symbols carry no duration, so repetition is the only available measure of importance.

Two part tags in the corpus use a nonstandard name (`<intro_riff_1>`, `<intro_riff_2>`). `PART_RE` expects `<name_number>` and does not match them, which is why the test is on `<`…`>` rather than on the regex. They are kept as tags (and reported in `unknown_parts`) instead of being mistaken for chords.

---

## 4. Parsing a chord symbol

Every chord symbol has the form `<root><suffix>[/<bass>]`. A regular expression splits it into those three parts.

*Section 9 · cell 39 · L1–5*
```python
# Root = A–G plus an optional accidental. "s" is a sharp only when it does not start
# "sus": Csus4 = C + sus4, but Cssus4 = C# + sus4.
CHORD_RE = re.compile(
    r"^(?P<root>[A-G](?:s(?!us)|b)?)(?P<suffix>[^/]*)(?:/(?P<bass>[A-G][sb]?))?$"
)
```

Piece by piece:

| Regex piece | Meaning |
|---|---|
| `^` … `$` | the whole token must match, not just part of it |
| `(?P<root> … )` | a named group called `root` |
| `[A-G]` | one note letter |
| `(?:s(?!us)\|b)?` | optionally a sharp `s` **not followed by `us`**, or a flat `b` |
| `(?P<suffix>[^/]*)` | the suffix: any characters up to a `/` (may be empty) |
| `(?:/(?P<bass>[A-G][sb]?))?` | optionally a `/` followed by a bass note |

The `(?!us)` part ("not followed by `us`") resolves the one real ambiguity of the spelling. Because `s` means sharp, `Csus4` could be read as C♯ + `us4`. The rule says an `s` that starts `sus` is part of the suffix:

| Token | root | suffix | bass |
|---|---|---|---|
| `Csus4` | `C` | `sus4` | – |
| `Cssus4` | `Cs` | `sus4` | – |
| `Fsmin` | `Fs` | `min` | – |
| `A/Cs` | `A` | `""` | `Cs` |
| `Cmin7/Bb` | `C` | `min7` | `Bb` |
| `Cs/` | *no match*: `/` with no bass note | | |
| `sC` | *no match*: does not start with A–G | | |

*Section 9 · cell 39 · L28–33*
```python
def parse_chord(token: str) -> tuple[str, str, str | None] | None:
    # (root, suffix, bass) or None when the token is not a well-formed chord.
    m = CHORD_RE.match(token)
    if m is None:
        return None
    return m.group("root"), m.group("suffix"), m.group("bass")
```

`parse_chord("C/E")` returns `("C", "", "E")`; `parse_chord("Gs/")` returns `None`.

---

## 5. Chord quality

### 5.1 The suffix table

Each of the 56 suffixes found in the corpus is mapped by hand to one quality.

*Section 9 · cell 39 · L7–22*
```python
# Every suffix that occurs in the corpus, mapped by hand. Anything else is "unknown".
SUFFIX_QUALITY = {
    **dict.fromkeys(["", "add9", "add11", "add13"], "major"),
    # minmaj* (minor triad + major 7th) is too rare (<10k tokens) for its own bucket.
    **dict.fromkeys(["min", "minadd9", "minadd11", "minadd13",
                     "minmaj7", "minmaj9", "minmaj11", "minmaj13"], "minor"),
    **dict.fromkeys(["7", "9", "11", "13", "7b9", "13b", "13b9", "11s", "11b9"], "dominant7"),
    **dict.fromkeys(["maj7", "maj9", "maj11", "maj13", "majs9", "maj911s", "majs911s", "maj1311s"], "major7"),
    **dict.fromkeys(["min7", "min9", "min11", "min13", "minb9", "min1113b"], "minor7"),
    **dict.fromkeys(["sus2", "sus4", "7sus2", "7sus4", "maj7sus2", "maj7sus4"], "suspended"),
    "no3d": "power",  # root + fifth, no third
    **dict.fromkeys(["dim", "dim7", "dim9", "dim11", "dimb9", "dimadd11", "dimadd13",
                     "dim11b9", "dim13b9"], "diminished"),
    "dimb7": "half-diminished",  # diminished triad + minor 7th (= m7b5)
    **dict.fromkeys(["aug", "augmaj7", "augmaj9", "augmaj11"], "augmented"),
}
```

`dict.fromkeys(list, value)` builds `{item: value}` for every item in the list, and `**` merges those small dictionaries into one. Written out, the table is simply `{"": "major", "add9": "major", …, "min": "minor", …}`.

Two mappings are musical decisions rather than obvious reads:

- `dimb7` → **half-diminished**. A diminished triad plus a minor 7th is m7♭5. The corpus never writes `m7b5` or `hdim`, which is why an earlier substring-based parser found zero half-diminished chords.
- `minmaj*` → **minor**. A minor triad with a major 7th is rare (under 10k tokens) and is grouped with the minor triads.

### 5.2 The lookup

*Section 9 · cell 39 · L36–40*
```python
def chord_quality(token: str) -> str:
    parsed = parse_chord(token)
    if parsed is None:
        return "unknown"
    return SUFFIX_QUALITY.get(parsed[1], "unknown")
```

- `parsed[1]` is the suffix.
- `.get(suffix, "unknown")` returns the quality, or `"unknown"` when the suffix is not in the table. **Nothing defaults to `major`.** A token that does not parse at all is also `"unknown"`.

The resulting qualities are: `major`, `minor`, `dominant7`, `major7`, `minor7`, `suspended`, `power`, `diminished`, `half-diminished`, `augmented`, `unknown`. A slash chord takes the quality of the chord to the left of `/`; the bass does not change it.

### 5.3 Validation

The notebook checks the parser in three ways:

1. **Hand-picked cases** (*Section 9 · cell 39 · L44–51*), asserting the ambiguous spellings: `Csus4` vs `Cssus4`, slash chords, power chords, `dimb7`, and the malformed `Cs/` and `sC`.
2. **Coverage of the whole vocabulary** (the cell after the parser). Every one of the 56 suffixes is in the table. Only **1,563 tokens (0.003%)** are `unknown`: 1,562 are slash chords with an empty bass (`Cs/`, `Gs7/`), one is `sC`.
3. **Comparison with the old heuristic** (the next cell). The earlier substring-based rules classified 2.68M tokens as major (7.2% of that bucket) that are dominant or major 7ths, and folded 1.41M minor 7ths into `minor`.

---

## 6. From text to numbers

### 6.1 Pitch classes

To compute intervals, note names must become numbers. Each root becomes its **pitch class**: semitones above C, from 0 to 11.

*Section 10 · cell 48 · L3–4*
```python
PITCH_CLASS = {"C": 0, "Cs": 1, "Db": 1, "D": 2, "Ds": 3, "Eb": 3, "E": 4, "F": 5, "Fs": 6,
               "Gb": 6, "G": 7, "Gs": 8, "Ab": 8, "A": 9, "As": 10, "Bb": 10, "B": 11}
```

Enharmonic spellings merge here: `Cs` and `Db` are both 1. The corpus uses both spellings for the same pitch.

### 6.2 `chord_pc`: one chord → (pitch class, quality)

*Section 10 · cell 48 · L22–28*
```python
@lru_cache(maxsize=None)  # ~4k distinct spellings, so each is parsed once
def chord_pc(token: str) -> tuple[int, str] | None:
    # (root pitch class, quality), or None for unknown tokens.
    quality = chord_quality(token)
    if quality == "unknown":
        return None
    return PITCH_CLASS[parse_chord(token)[0]], quality
```

| Line | What it does | For `"C/E"` |
|---|---|---|
| `@lru_cache(maxsize=None)` | remembers each answer; the loop calls this ~27M times but there are only 4,314 spellings | — |
| `chord_quality(token)` | parse + suffix lookup (Section 5) | `"major"` |
| `if quality == "unknown": return None` | malformed tokens stop here | not unknown |
| `PITCH_CLASS[parse_chord(token)[0]]` | `[0]` is the root text, `"C"` → `0` | returns `(0, "major")` |

**The bass note is dropped here.** `C/E` and `C` become the same pair.

### 6.3 Applying it to the song

*Section 10 · cell 49 · L11–14*
```python
    ctoks, _ = split_tokens(raw)
    chords = [c for c in map(chord_pc, ctoks) if c is not None]
    if not chords:
        continue
```

- `map(chord_pc, ctoks)` runs `chord_pc` on every token.
- `[c for c in … if c is not None]` keeps only the successful results, removing unknown tokens.
- `if not chords: continue` skips a song with no usable chord.

| token | `chord_pc(token)` | kept? |
|---|---|---|
| `C` | `(0, "major")` | ✓ |
| `G` | `(7, "major")` | ✓ |
| `Amin` | `(9, "minor")` | ✓ |
| `F` | `(5, "major")` | ✓ |
| `F` | `(5, "major")` | ✓ |
| `C/E` | `(0, "major")` | ✓ |
| `Gs/` | `None` | ✗ |
| `G` | `(7, "major")` | ✓ |

This list of 7 pairs is what `estimate_key` receives.

### 6.4 Triad types

For the key test, the detailed quality is reduced to a triad type.

*Section 10 · cell 48 · L8–12*
```python
# Triad type used to test whether a chord belongs to a key. Power and suspended chords
# have no third, so only their root is tested ("any").
TRIAD = {"major": "maj", "dominant7": "maj", "major7": "maj", "minor": "min", "minor7": "min",
         "diminished": "dim", "half-diminished": "dim", "augmented": "aug",
         "suspended": "any", "power": "any"}
```

| Qualities | Triad type | Why |
|---|---|---|
| major, dominant7, major7 | `maj` | a major triad with or without a 7th |
| minor, minor7 | `min` | a minor triad with or without a 7th |
| diminished, half-diminished | `dim` | a diminished triad with or without a 7th |
| augmented | `aug` | never diatonic to a major key |
| suspended, power | `any` | no third, so it cannot be classed as major or minor |

---

## 7. Key estimation

### 7.1 The idea

A major key is a family of seven chords: I, ii, iii, IV, V, vi, vii°. The estimate tries all 12 major keys and picks the one whose family contains **the most chord tokens of the song**. It is template matching, like guessing a sentence's language by counting how many of its words are in each dictionary.

### 7.2 One template for all 12 keys

The code does not store 12 families. It stores **one**, written as distances from the tonic.

*Section 10 · cell 48 · L17–19*
```python
# Diatonic triads of a major key as (semitones above the tonic, triad type).
DIATONIC = {(0, "maj"), (2, "min"), (4, "min"), (5, "maj"), (7, "maj"), (9, "min"), (11, "dim")}
SCALE = {degree for degree, _ in DIATONIC}
```

| pair in `DIATONIC` | `(0,maj)` | `(2,min)` | `(4,min)` | `(5,maj)` | `(7,maj)` | `(9,min)` | `(11,dim)` |
|---|---|---|---|---|---|---|---|
| degree | I | ii | iii | IV | V | vi | vii° |

`SCALE` keeps only the first number of each pair: `{0, 2, 4, 5, 7, 9, 11}`, the major scale in semitones. It is derived from `DIATONIC` so the two can never disagree. Both are **sets**, because the only question ever asked of them is "is this in there?".

### 7.3 The shift: how one template serves every key

The chord is moved, not the template. To test a chord against key `key`, the caller subtracts the tonic:

```python
(pc - key) % 12
```

The `% 12` (remainder after dividing by 12) wraps around the octave like a clock: `(0 − 7) % 12 = 5`.

| Question | Calculation | Distance | In the template? |
|---|---|---|---|
| Is G in C major? | `(7 − 0) % 12` | 7 | `(7, "maj")` ✓ → V |
| Is C in G major? | `(0 − 7) % 12` | 5 | `(5, "maj")` ✓ → IV |
| Is Am in G major? | `(9 − 7) % 12` | 2 | `(2, "min")` ✓ → ii |
| Is F in G major? | `(5 − 7) % 12` | 10 | `(10, "maj")` ✗ |

"Is Am the ii of G?" becomes "is a minor chord 2 semitones above the tonic in the major template?". It is the same question with the key subtracted out.

### 7.4 `is_diatonic`: one chord against the template

*Section 10 · cell 48 · L31–32*
```python
def is_diatonic(degree: int, triad: str) -> bool:
    return degree in SCALE if triad == "any" else (degree, triad) in DIATONIC
```

The one-liner uses Python's `A if condition else B`. Written longhand:

```python
def is_diatonic(degree, triad):
    if triad == "any":                        # power or sus chord: no third to compare
        return degree in SCALE                # only the root must be on a scale degree
    else:                                     # maj, min, dim, aug
        return (degree, triad) in DIATONIC    # position AND chord type must match
```

`is_diatonic` **never sees the key**. It receives a distance that the caller already computed, and checks it against the C-major-shaped template.

| chord (key = C) | `degree` | `triad` | check | result |
|---|---|---|---|---|
| G | 7 | maj | `(7, "maj") in DIATONIC` | ✓ V |
| Am | 9 | min | `(9, "min") in DIATONIC` | ✓ vi |
| Bb | 10 | maj | `(10, "maj") in DIATONIC` | ✗ ♭VII |
| D | 2 | maj | `(2, "maj") in DIATONIC` | ✗ right root, wrong type (the key has Dm) |
| E5 (power) | 4 | any | `4 in SCALE` | ✓ |
| F♯5 (power) | 6 | any | `6 in SCALE` | ✗ |
| Caug | 0 | aug | `(0, "aug") in DIATONIC` | ✗ augmented never fits |

The D row shows why the template stores pairs and not only the scale: a chord must match both its position and its type.

### 7.5 `estimate_key`, line by line

*Section 10 · cell 48 · L35–45*
```python
def estimate_key(chords: list[tuple[int, str]]) -> tuple[int, float]:
    # Major key (tonic pitch class) with the most diatonic chord tokens, and the key fit.
    counts = Counter((pc, TRIAD[q]) for pc, q in chords)
    best_score, best_key = None, 0
    for key in range(12):
        diatonic = sum(n for (pc, t), n in counts.items() if is_diatonic((pc - key) % 12, t))
        home = counts[(key, "maj")] + counts[((key + 9) % 12, "min")]  # I and vi chords
        score = (diatonic, home, -key)  # -key only makes the last tie-break deterministic
        if best_score is None or score > best_score:
            best_score, best_key = score, key
    return best_key, best_score[0] / len(chords)
```

**L37: count the chords.** Each chord is reduced to (pitch class, triad type), and `Counter` counts the occurrences:

```python
counts = {(0, "maj"): 2,   # C, C/E
          (7, "maj"): 2,   # G, G
          (9, "min"): 1,   # Am
          (5, "maj"): 2}   # F, F
```

The 7 chords collapse to 4 distinct ones, so each distinct chord is tested once per key.

**L38: nothing is best yet.** `best_score = None` until the first key has been scored.

**L39: try each key.** `range(12)` gives `key = 0 … 11` (C, D♭, D, … B). L40–44 run once per key.

**L40: count the fitting tokens.** The same line as an ordinary loop:

```python
diatonic = 0
for (pc, t), n in counts.items():         # e.g. pc=5, t="maj", n=2   (F ×2)
    degree = (pc - key) % 12              # distance from this candidate tonic
    if is_diatonic(degree, t):            # does it fit the template?
        diatonic += n                     # yes → add how many times it was played
```

`(pc, t), n` unpacks each counter item, which looks like `((5, "maj"), 2)`, into three variables.

**L41: the tie-break value.** `home` = how often the song plays this key's I chord (major on the tonic) plus its vi chord. `(key + 9) % 12` is the root 9 semitones above the tonic, the relative-minor tonic. Songs keep returning to their home chord, so among tied keys the more "central" one wins.

**L42: the score.** Everything is packed into a tuple `(diatonic, home, -key)`.

**L43–44: keep the best.** Python compares tuples **left to right**:
- `(7, 3, 0) > (5, 2, -5)` because 7 > 5; the other elements are not looked at.
- `home` only decides when `diatonic` ties.
- `-key` only decides when both tie. It prefers the lowest pitch class and has no musical meaning; it only makes the result deterministic.
- `best_score is None` handles the first key, when there is nothing to compare against.

**L45: return.** The winning key, and the **key fit** = diatonic tokens ÷ all chord tokens.

### 7.6 Full trace of the running example

Each cell shows `(pc − key) % 12` and the triad type, then whether it is in the template.

| key | C (×2) | G (×2) | Am (×1) | F (×2) | `diatonic` | `home` (I + vi) | `score` |
|---|---|---|---|---|---|---|---|
| 0 = C | 0,maj ✓ | 7,maj ✓ | 9,min ✓ | 5,maj ✓ | **7** | C(2) + Am(1) = 3 | `(7, 3, 0)` |
| 5 = F | 7,maj ✓ | 2,maj ✗ | 4,min ✓ | 0,maj ✓ | 5 | F(2) + Dm(0) = 2 | `(5, 2, -5)` |
| 7 = G | 5,maj ✓ | 0,maj ✓ | 2,min ✓ | 10,maj ✗ | 5 | G(2) + Em(0) = 2 | `(5, 2, -7)` |
| 2 = D | 10,maj ✗ | 5,maj ✓ | 7,min ✗ | 3,maj ✗ | 2 | D(0) + Bm(0) = 0 | `(2, 0, -2)` |

The other eight keys score lower. Result: `key = 0` (C major), key fit = 7 / 7 = **1.0**.

### 7.7 When the tie-break matters

Neighbouring keys share six of their seven chords, so a song that avoids the chord that tells them apart ties on `diatonic`. For **C ×4, G ×3, Am ×3**:

| key | diatonic | home |
|---|---|---|
| C major | C=I, G=V, Am=vi → 10 | C(4) + Am(3) = **7** |
| G major | C=IV, G=I, Am=ii → 10 | G(3) + Em(0) = 3 |

Both fit completely, and C wins on `home`. If the song had one F chord, C would win on `diatonic` alone, because G major has F♯.

### 7.8 Minor keys

**Natural minor needs nothing special.** A minor key uses the same seven chords as its relative major:

| A natural minor | Am | Bdim | C | Dm | Em | F | G |
|---|---|---|---|---|---|---|---|
| as C major | vi | vii° | I | ii | iii | IV | V |

A song in A minor fits the C-major template completely. For `Am F C G Am`:

| key | Am | F | C | G | Am | fits |
|---|---|---|---|---|---|---|
| 0 (C) | 9,min ✓ | 5,maj ✓ | 0,maj ✓ | 7,maj ✓ | ✓ | **5 / 5** |
| 9 (A major) | 0,min ✗ | 8,maj ✗ | 3,maj ✗ | 10,maj ✗ | ✗ | 0 / 5 |

The estimate is C with fit 1.0, and the numerals are `vi IV I V vi`. **The code never decides between major and minor.** It always reports the relative major, and a minor tonic appears as `vi`. This is deliberate: the two share the same chords, so the template cannot tell them apart. For comparing progressions, `vi IV I V` carries the same information as `i VI III VII`. This is also why `home` counts the vi chord.

**Harmonic minor lowers the fit.** Minor songs often play a major V (E instead of Em in A minor), which is not in the major template. For `Am Dm E Am`:

| key | Am (×2) | Dm | E | fits | home |
|---|---|---|---|---|---|
| 0 (C) | 9,min ✓ | 2,min ✓ | 4,maj ✗ (template has `(4,"min")`) | 3 / 4 | C(0) + Am(2) = **2** |
| 5 (F) | 4,min ✓ | 9,min ✓ | 11,maj ✗ (template has `(11,"dim")`) | 3 / 4 | F(0) + Dm(1) = 1 |

C and F tie at 3/4, and `home` picks C because the minor tonic Am appears twice. The key is right, but the **fit drops to 0.75**, and the E chord comes out as `III` (read it as V/vi). Minor songs with a raised leading tone therefore get lower key fits and depend more on the tie-break.

### 7.9 Key fit as a confidence measure

The key fit is the share of a song's chord tokens that are diatonic to the estimated key. It is stored per song:

*Section 10 · cell 49 · L15–16*
```python
    key, fit = estimate_key(chords)
    key_est[i], key_fit[i] = key, fit
```

A low fit means one of three things: the song modulates, it uses borrowed or chromatic chords, or the estimate is unreliable. The fit cannot tell these apart, so it is treated both as a quality signal and as a candidate feature (Section 11).

---

## 8. Roman numerals

Once the key is known, each chord is named by its distance from the tonic.

*Section 10 · cell 48 · L6, L13–15*
```python
DEGREE = ["I", "bII", "II", "bIII", "III", "IV", "#IV", "V", "bVI", "VI", "bVII", "VII"]
```
```python
NUMERAL_SUFFIX = {"major": "", "minor": "", "dominant7": "7", "major7": "maj7", "minor7": "7",
                  "diminished": "°", "half-diminished": "ø", "augmented": "+",
                  "suspended": "sus", "power": "5"}
```

`DEGREE` has one name per distance, positions 0 to 11:

| distance | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| name | I | ♭II | II | ♭III | III | IV | ♯IV | V | ♭VI | VI | ♭VII | VII |

*Section 10 · cell 48 · L48–52*
```python
def to_numeral(pc: int, quality: str, key: int) -> str:
    numeral = DEGREE[(pc - key) % 12]
    if TRIAD[quality] in ("min", "dim"):
        numeral = numeral.lower()
    return numeral + NUMERAL_SUFFIX[quality]
```

| Line | What it does | Bm7 in key A (`pc=11`, `key=9`) |
|---|---|---|
| `DEGREE[(pc - key) % 12]` | distance → name | `(11 − 9) % 12 = 2` → `"II"` |
| `if TRIAD[quality] in ("min", "dim")` | minor and diminished → lower case | minor7 → `"min"` → `"ii"` |
| `+ NUMERAL_SUFFIX[quality]` | adds the extension | `"7"` → `"ii7"` |

Chords outside the key come out chromatic: B♭ in C → `bVII`, E major in C → `III`, D major in C → `II`.

Applied to every chord of the song:

*Section 10 · cell 49 · L17*
```python
    rn = [to_numeral(pc, q, key) for pc, q in chords]
```

Running example, key C: `["I", "V", "vi", "IV", "IV", "I", "V"]`.

**Why this is transposition-invariant.** The same song in D (`D A Bmin G G D/Fs A`) gets key 2. Every chord's pitch class is also 2 higher, so every distance `(pc − key) % 12` is unchanged, and so is every numeral. The notebook asserts this on the reviewer's example.

*Section 10 · cell 48 · L61, L71–72*
```python
examples = ["C G Amin F", "D A Bmin G", "Amin F C G", "E B Csmin A", "G D Emin7 C/E"]
```
```python
assert examples_df.loc[[0, 1, 3], "roman"].eq("I V vi IV").all()
assert examples_df.loc[2, "roman"] == "vi IV I V"
```

| progression | estimated key | key fit | roman | intervals |
|---|---|---|---|---|
| `C G Amin F` | C | 1.0 | `I V vi IV` | `+7maj +2min +8maj` |
| `D A Bmin G` | D | 1.0 | `I V vi IV` | `+7maj +2min +8maj` |
| `Amin F C G` | C | 1.0 | `vi IV I V` | `+8maj +7maj +7maj` |
| `E B Csmin A` | E | 1.0 | `I V vi IV` | `+7maj +2min +8maj` |
| `G D Emin7 C/E` | G | 1.0 | `I V vi7 IV` | `+7maj +2min +8maj` |

---

## 9. Intervals: the key-free representation

Roman numerals depend on the key estimate, so a wrong key gives wrong numerals. Intervals avoid the key entirely: each token is the root movement from one chord to the next, plus the triad type of the chord arrived at.

*Section 10 · cell 48 · L55–57*
```python
def to_intervals(chords: list[tuple[int, str]]) -> list[str]:
    # Semitones from each chord root to the next, plus the triad type of the arriving chord.
    return [f"+{(b[0] - a[0]) % 12}{TRIAD[b[1]]}" for a, b in zip(chords, chords[1:])]
```

- `zip(chords, chords[1:])` pairs each chord with the next: (1st, 2nd), (2nd, 3rd), …
- `(b[0] - a[0]) % 12` is the upward root distance in semitones.
- `TRIAD[b[1]]` is the triad type of the arriving chord.

Running example:

| from → to | calculation | token |
|---|---|---|
| C → G | (7 − 0) % 12 | `+7maj` |
| G → Am | (9 − 7) % 12 | `+2min` |
| Am → F | (5 − 9) % 12 | `+8maj` |
| F → F | (5 − 5) % 12 | `+0maj` |
| F → C/E | (0 − 5) % 12 | `+7maj` |
| C/E → G | (7 − 0) % 12 | `+7maj` |

**Trade-off:** intervals cannot be wrong about the key, but they lose harmonic function. `+7maj` is "up a fifth to a major chord" whether that means I→V or IV→I. The notebook uses them as a **cross-check**: a conclusion that holds under both Roman numerals and intervals does not depend on key-estimation errors.

---

## 10. Four-chord windows and per-genre counts

Both relative representations are cut into windows that span four consecutive chords and counted per genre.

*Section 10 · cell 49 · L19–23*
```python
    for j in range(len(chords) - 3):
        if chords[j] == chords[j + 2] and chords[j + 1] == chords[j + 3]:
            continue
        quads_rn_by_genre[genre][tuple(rn[j:j + 4])] += 1
        quads_iv_by_genre[genre][tuple(iv[j:j + 3])] += 1  # 3 intervals span the same 4 chords
```

| Line | What it does |
|---|---|
| `range(len(chords) - 3)` | 7 chords → 4 windows, `j = 0, 1, 2, 3` |
| `if chords[j] == chords[j + 2] and …` | skips A-B-A-B windows (`C G C G`), which only reflect vamping on two chords. The same rule is used for the absolute 4-grams (*Section 9 · cell 44 · L23*) |
| `rn[j:j + 4]` | four numerals; `tuple(...)` makes the window usable as a dictionary key |
| `iv[j:j + 3]` | three intervals connect the same four chords |
| `quads_rn_by_genre[genre][…] += 1` | adds one to that pattern's count for the song's genre |

`quads_rn_by_genre` is a `defaultdict(Counter)`: one counter per genre, created automatically the first time a genre appears. The absolute counterpart, `quads_by_genre`, is built the same way from the raw tokens (*Section 9 · cell 44 · L19–25*). It keeps the bass notes and the unknown tokens, and checks ABAB on the spellings.

Windows counted for the running example:

| j | Roman numerals | intervals |
|---|---|---|
| 0 | `I V vi IV` | `+7maj +2min +8maj` |
| 1 | `V vi IV IV` | `+2min +8maj +0maj` |
| 2 | `vi IV IV I` | `+8maj +0maj +7maj` |
| 3 | `IV IV I V` | `+0maj +7maj +7maj` |

These counters feed the vocabulary table, the top patterns per genre, and the cosine-similarity heatmaps of Section 10. The per-song key and fit go into `lab_keys` (*Section 10 · cell 49 · L25–27*) for the key-fit and key-by-genre plots.

---

## 11. What the representations show on the corpus

Figures from the full run on the 352,111 labeled tracks.

**Key estimate**

| Measure | Value |
|---|---|
| Median key fit | 0.97 |
| Tracks with fit ≥ 0.9 | 64.5% |
| Tracks with fit ≥ 0.7 | 88.7% |
| Lowest-fit genres (share of tracks below 0.8) | jazz 35.6%, metal 32.6%, soul 29.6% |
| Highest-fit genres | country 13.2%, electronic 13.7%, reggae 15.3% |

**Vocabulary of four-chord patterns** (same ~23.3M windows)

| Representation | Distinct patterns | Top-100 coverage | Top-1000 coverage |
|---|---|---|---|
| Absolute chords | 1,365,227 | 15.9% | 41.2% |
| Roman numerals | 610,845 | 35.9% | 63.3% |
| Intervals | 51,029 | 46.2% | 84.1% |

Transposition merges equivalent progressions: the number of distinct patterns falls by more than half, and the most common patterns cover much more of the data.

**Most common Roman-numeral patterns** make genre habits readable: rotations of `I V vi IV` in pop, rap, reggae and electronic; the three-chord `I V IV` loop in rock, pop rock, alternative and punk; `I IV I V` in country; loops centered on the relative minor (`vi IV V`, `vi IV I V`) in metal.

**Genre-by-genre similarity** of the top patterns barely changes with the representation: mean off-diagonal cosine similarity is 0.871 (absolute), 0.876 (Roman) and 0.864 (intervals). The pop/rock cluster is therefore not an artifact of key.

**The key itself** is associated with genre, which is why it is kept as a separate feature rather than thrown away. Country is 3.4% in flat keys (D♭, E♭, A♭, B♭), while soul, reggae, rap and jazz are at 13–17%. Guitar tabs are often written in capo shapes, so the written key is not always the sounding key.

---

## 12. Known limitations and possible improvements

| Limitation | Effect | Possible fix |
|---|---|---|
| **One key per song** | A song that modulates gets a single compromise key and a lower fit | Estimate per section (`split_tokens` would need to return sections) or on a sliding window |
| **Major template only** | Minor songs are written relative to the relative major (`vi` = tonic). This is intentional | Add minor templates if `i`-based numerals are ever needed |
| **Harmonic-minor V is not diatonic** | Minor songs with a major V get lower fit and rely on the tie-break (7.8) | Add `(4, "maj")`, i.e. III = V/vi, to `DIATONIC`, then compare fit per genre before and after |
| **Power-chord-only songs** | Power chords are type `any` and never add to `home`. `E5 G5 A5` fits C, D, F and G equally, ties on `home` = 0, and the arbitrary `-key` tie-break picks C. The riff is really in E minor (relative major G). The fit is still 1.0, so it is not flagged | Let `any` chords count towards `home`, or add the first/last chord of the song as a tie-break |
| **Bass notes dropped** | `A/C♯` and `A` both become `I`; inversions are invisible to the relative representations | Add the bass as a degree too, e.g. `I/3` |
| **Windows cross section boundaries** | Part tags are removed before windowing, so a window can span the end of a verse and the start of a chorus | Window within each section |
| **Removed unknown tokens join their neighbours** | Dropping `Gs/` puts `C/E` directly before `G` | Negligible: unknown tokens are 0.003% of the data |
| **Chord counts, no durations** | A chord's weight in the key estimate is how often it is written, not how long it sounds | None available: the corpus has no durations |

**Compared with standard methods.** The classic key-finding algorithm is Krumhansl–Schmuckler. It expands chords into notes and correlates the song's note distribution with empirical profiles for all 24 major and minor keys. The method here is a simplified version of the same idea: chords instead of notes, a yes/no template instead of weighted profiles, 12 keys instead of 24. It is easier to explain and to check, and it is precise enough when the key fit is high, which is the case for most of the corpus.

---

## 13. Quick reference

| Name | Kind | Location | Role |
|---|---|---|---|
| `labeled_chords`, `labeled_genres` | arrays | Section 9 · cell 44 · L4–5 | chord strings and genres of the 352,111 labeled songs |
| `PART_RE`, `split_tokens` | regex, function | Section 8 · cell 33 · L1–16 | separate part tags from chord tokens |
| `CHORD_RE`, `parse_chord` | regex, function | Section 9 · cell 39 · L3–5, L28–33 | token → (root, suffix, bass) |
| `SUFFIX_QUALITY`, `chord_quality` | dict, function | Section 9 · cell 39 · L8–22, L36–40 | suffix → quality, `unknown` fallback |
| `PITCH_CLASS` | dict | Section 10 · cell 48 · L3–4 | root name → 0–11 |
| `TRIAD` | dict | Section 10 · cell 48 · L10–12 | quality → triad type for the key test |
| `chord_pc` | function (cached) | Section 10 · cell 48 · L22–28 | token → (pitch class, quality) or `None` |
| `DIATONIC`, `SCALE` | sets | Section 10 · cell 48 · L18–19 | the major-key template |
| `is_diatonic` | function | Section 10 · cell 48 · L31–32 | does a chord at this distance fit the template? |
| `estimate_key` | function | Section 10 · cell 48 · L35–45 | chords → (key, key fit) |
| `DEGREE`, `NUMERAL_SUFFIX` | list, dict | Section 10 · cell 48 · L6, L13–15 | distance → numeral name; quality → extension |
| `to_numeral` | function | Section 10 · cell 48 · L48–52 | (chord, key) → Roman numeral |
| `to_intervals` | function | Section 10 · cell 48 · L55–57 | chords → interval tokens |
| `quads_rn_by_genre`, `quads_iv_by_genre` | per-genre counters | Section 10 · cell 49 · L6–7, L19–23 | four-chord pattern counts |
| `lab_keys` | DataFrame | Section 10 · cell 49 · L25–27 | estimated key and key fit per labeled song |
