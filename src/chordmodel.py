"""Pipeline del predictor de genero a partir de `chords`.

Todo lo que el modelo guardado necesita para existir por si solo: constantes,
lectura de acordes, transformadores y fabricas de pipeline. Los notebooks 02 y
03 importan de aqui. Como las clases viven en este modulo y no en `__main__`,
un `model.joblib` se carga desde cualquier script con `src/` en `sys.path`.

Ninguna funcion de este modulo ajusta con datos. Las fabricas entregan
estimadores sin fit.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------

SEED = 42
NGRAM_ORDERS = (1, 2, 3, 4)
NGRAM_MIN_DF = 5
NGRAM_MAX_FEATURES = 5000

MAIN_GENRES = frozenset(
    {
        "pop",
        "rock",
        "country",
        "alternative",
        "pop rock",
        "punk",
        "metal",
        "rap",
        "soul",
        "jazz",
        "reggae",
        "electronic",
    }
)
LABELS = sorted(MAIN_GENRES)

# Identidad, fuente de la etiqueta, otra tarea, tiempo y el target.
EXCLUDED_COLUMNS = [
    "id",
    "artist_id",
    "spotify_song_id",
    "spotify_artist_id",
    "genres",
    "rock_genre",
    "release_date",
    "decade",
    "main_genre",
    "chords",
]

PART_RE = re.compile(r"^<([A-Za-z]+)_(\d+)>$")
POWER_RE = re.compile(r"^[A-G]s?(?:no3|5)(?:d)?$", re.I)
_SUFFIX_CHUNKS = (
    "minmaj",
    "augmaj",
    "min7b5",
    "m7b5",
    "hmin",
    "hdim",
    "min",
    "maj",
    "dim",
    "aug",
    "sus",
    "add",
    "no3",
    "13",
    "11",
    "9",
    "7",
    "6",
    "5",
    "4",
    "2",
    "b9",
    "b7",
    "b",
    "s",
    "d",
)

# El medio disminuido se reconoce, como en el EDA, y no se emite:
# alli esas grafias no aparecen, asi que la columna seria constante.
QUALITY_COLUMNS = (
    "share_major",
    "share_minor",
    "share_power",
    "share_suspended",
    "share_diminished",
    "share_augmented",
    "share_unknown",
)
_QUALITY_INDEX = {
    "major": 0,
    "minor": 1,
    "power": 2,
    "suspended": 3,
    "diminished": 4,
    "augmented": 5,
    "unknown": 6,
}

NUMERIC_FEATURES = [
    "n_chords",
    "log1p_n_chords",
    "n_unique_chords",
    "unique_ratio",
    *QUALITY_COLUMNS,
    "abab_share",
]
BINARY_FEATURES = ["has_structure"]
FEATURE_COLUMNS = NUMERIC_FEATURES + BINARY_FEATURES

# Krumhansl & Kessler (1982), perfiles en do mayor y do menor.
# La misma rotacion que en la seccion 10 del EDA.
KS_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
KS_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
_KS_TEMPLATES = np.stack(
    [np.roll(KS_MAJOR, shift) for shift in range(12)]
    + [np.roll(KS_MINOR, shift) for shift in range(12)]
)
_KS_CENTERED = _KS_TEMPLATES - _KS_TEMPLATES.mean(axis=1, keepdims=True)
_KS_NORMS = np.linalg.norm(_KS_CENTERED, axis=1)

_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
# Semitonos desde la tonica hacia los grados 1-7. Los cinco intervalos
# cromaticos se pliegan sobre el grado vecino, como en el EDA.
_FOLD = (1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7)

# --------------------------------------------------------------------------
# Lectura de acordes
# --------------------------------------------------------------------------


def _recognized_suffix(rest: str) -> bool:
    # Misma lectura que el EDA: un resto no reconocido no se llama mayor.
    text = rest.lower()
    while text:
        for chunk in _SUFFIX_CHUNKS:
            if text.startswith(chunk):
                text = text[len(chunk) :]
                break
        else:
            return False
    return True


def chord_quality(token: str) -> str:
    # Raiz A-G, accidente s/b (sus no es sostenido). El slash usa el acorde a la izquierda.
    core = token.split("/", 1)[0]
    if not core or core[0] not in "ABCDEFG":
        return "unknown"
    rest = core[1:]
    if rest and rest[0] in "bB":
        rest = rest[1:]
    elif rest and rest[0] in "sS" and not rest[1:].lower().startswith("us"):
        rest = rest[1:]
    if not _recognized_suffix(rest):
        return "unknown"
    lowered = core.lower()
    if "sus" in lowered:
        return "suspended"
    if "aug" in lowered:
        return "augmented"
    if "hmin" in lowered or "hdim" in lowered or "min7b5" in lowered or "m7b5" in lowered:
        return "half-diminished"
    if "dim" in lowered:
        return "diminished"
    if "no3" in lowered or POWER_RE.match(core):
        return "power"
    if "min" in lowered:
        return "minor"
    return "major"


def _chords_missing(raw) -> bool:
    if raw is None:
        return True
    missing = pd.isna(raw)
    if isinstance(missing, np.ndarray):
        return False
    return bool(missing)


def _require_chords_frame(X, name: str) -> None:
    if not isinstance(X, pd.DataFrame) or "chords" not in X.columns:
        raise ValueError(f"{name} necesita un DataFrame con la columna 'chords'.")


def chord_tokens(raw) -> tuple[list[str], float]:
    """Acordes y bandera de forma. Un nulo no se lee como el texto 'nan'."""
    if _chords_missing(raw):
        raise ValueError(
            "chords esta ausente. El transformador no convierte un valor nulo en el texto 'nan'."
        )
    tokens: list[str] = []
    has_structure = 0.0
    for token in str(raw).split():
        if PART_RE.match(token):
            has_structure = 1.0
        else:
            tokens.append(token)
    return tokens, has_structure


def root_pc(token: str) -> int | None:
    """Clase de altura de la triada a la izquierda de `/`. None si la fundamental no es A-G."""
    core = token.split("/", 1)[0]
    if not core or core[0] not in _PC:
        return None
    rest = core[1:]
    acc = 0
    # Un "G" a secas tiene resto vacio y sigue siendo G; solo un accidente real lo mueve.
    if rest and rest[0] in "bB":
        acc = -1
    elif rest and rest[0] in "sS" and not rest[1:].lower().startswith("us"):
        acc = 1
    return (_PC[core[0]] + acc) % 12


def degree_tokens(tokens: list[str]) -> list[str]:
    """Grado y cualidad respecto de la tonalidad estimada en esta misma pista."""
    if not tokens:
        return []
    hist = np.zeros(12, dtype=np.float64)
    pcs: list[int] = []
    qualities: list[str] = []
    for token in tokens:
        pc = root_pc(token)
        if pc is None:
            pcs.append(-1)
            qualities.append("other")
            continue
        hist[pc] += 1.0
        pcs.append(pc)
        qualities.append(chord_quality(token))
    if hist.sum() == 0:
        return ["other"] * len(tokens)
    centered = hist - hist.mean()
    norm = float(np.linalg.norm(centered))
    if norm == 0.0:
        return ["other"] * len(tokens)
    correlation = (centered @ _KS_CENTERED.T) / (norm * _KS_NORMS)
    tonic = int(np.argmax(correlation)) % 12
    degrees: list[str] = []
    for pc, quality in zip(pcs, qualities):
        if pc < 0:
            degrees.append("other")
        else:
            degree = _FOLD[(pc - tonic) % 12]
            degrees.append(f"{degree}:{quality}")
    return degrees


def _ngram_key(gram: tuple[str, ...]) -> str:
    return " ".join(gram)


# --------------------------------------------------------------------------
# Transformadores
# --------------------------------------------------------------------------


class ChordFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Features por pista a partir de `chords`. No estima nada en `fit`."""

    def fit(self, X, y=None):
        _require_chords_frame(X, "ChordFeaturesTransformer")
        for raw in X["chords"].to_numpy():
            chord_tokens(raw)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        _require_chords_frame(X, "ChordFeaturesTransformer")
        chords = X["chords"].to_numpy()
        n_rows = len(chords)
        out = np.zeros((n_rows, len(FEATURE_COLUMNS)), dtype=np.float64)
        i_n = FEATURE_COLUMNS.index("n_chords")
        i_log = FEATURE_COLUMNS.index("log1p_n_chords")
        i_unique = FEATURE_COLUMNS.index("n_unique_chords")
        i_ratio = FEATURE_COLUMNS.index("unique_ratio")
        i_abab = FEATURE_COLUMNS.index("abab_share")
        i_structure = FEATURE_COLUMNS.index("has_structure")
        i_quality = FEATURE_COLUMNS.index("share_major")

        for row, raw in enumerate(chords):
            tokens, has_structure = chord_tokens(raw)
            n_tokens = len(tokens)
            out[row, i_n] = n_tokens
            out[row, i_log] = np.log1p(n_tokens)
            out[row, i_unique] = len(set(tokens))
            out[row, i_ratio] = (len(set(tokens)) / n_tokens) if n_tokens else 0.0
            out[row, i_structure] = has_structure
            if n_tokens:
                counts = np.zeros(len(QUALITY_COLUMNS), dtype=np.float64)
                for token in tokens:
                    quality = _QUALITY_INDEX.get(chord_quality(token))
                    if quality is not None:
                        counts[quality] += 1.0
                out[row, i_quality : i_quality + len(QUALITY_COLUMNS)] = counts / n_tokens
            if n_tokens >= 4:
                windows = n_tokens - 3
                abab = 0
                for start in range(windows):
                    if (
                        tokens[start] == tokens[start + 2]
                        and tokens[start + 1] == tokens[start + 3]
                    ):
                        abab += 1
                out[row, i_abab] = abab / windows
        return pd.DataFrame(out, index=X.index, columns=FEATURE_COLUMNS)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURE_COLUMNS, dtype=object)


class ChordNgramTransformer(BaseEstimator, TransformerMixin):
    """Conteos de n-gramas. El vocabulario se aprende en `fit` y no crece en `transform`."""

    def __init__(self, n=2, alphabet="absolute", min_df=5, max_features=5000):
        self.n = n
        self.alphabet = alphabet
        self.min_df = min_df
        self.max_features = max_features

    def _check_params(self) -> None:
        if not isinstance(self.n, int) or isinstance(self.n, bool) or self.n < 1:
            raise ValueError("n debe ser un entero >= 1.")
        if self.alphabet not in ("absolute", "degree"):
            raise ValueError(
                f"alphabet debe ser 'absolute' o 'degree', no {self.alphabet!r}."
            )
        if not isinstance(self.min_df, int) or isinstance(self.min_df, bool) or self.min_df < 1:
            raise ValueError("min_df debe ser un entero >= 1.")
        if self.max_features is not None and (
            not isinstance(self.max_features, int)
            or isinstance(self.max_features, bool)
            or self.max_features < 1
        ):
            raise ValueError("max_features debe ser un entero >= 1.")

    def _sequence(self, raw) -> list[str]:
        tokens, _ = chord_tokens(raw)
        if self.alphabet == "absolute":
            return tokens
        return degree_tokens(tokens)

    def _ngrams(self, tokens: list[str]) -> list[tuple[str, ...]]:
        n = self.n
        if len(tokens) < n:
            return []
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def fit(self, X, y=None):
        _require_chords_frame(X, "ChordNgramTransformer")
        self._check_params()
        document_frequency: Counter[tuple[str, ...]] = Counter()
        for raw in X["chords"].to_numpy():
            document_frequency.update(set(self._ngrams(self._sequence(raw))))
        ranked = sorted(
            (
                (gram, frequency)
                for gram, frequency in document_frequency.items()
                if frequency >= self.min_df
            ),
            key=lambda item: (-item[1], _ngram_key(item[0])),
        )
        if self.max_features is not None:
            ranked = ranked[: self.max_features]
        self.vocabulary_ = {
            _ngram_key(gram): index for index, (gram, _) in enumerate(ranked)
        }
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        _require_chords_frame(X, "ChordNgramTransformer")
        self._check_params()
        sequences = [self._sequence(raw) for raw in X["chords"].to_numpy()]
        if not hasattr(self, "vocabulary_"):
            raise RuntimeError(
                "ChordNgramTransformer no esta ajustado. Llama a fit antes de transform."
            )
        n_rows = len(sequences)
        n_cols = len(self.vocabulary_)
        vocabulary = self.vocabulary_
        indptr = np.zeros(n_rows + 1, dtype=np.int64)
        indices: list[int] = []
        data: list[float] = []
        for row, tokens in enumerate(sequences):
            counts: dict[int, int] = {}
            for gram in self._ngrams(tokens):
                column = vocabulary.get(_ngram_key(gram))
                if column is not None:
                    counts[column] = counts.get(column, 0) + 1
            for column in sorted(counts):
                indices.append(column)
                data.append(float(counts[column]))
            indptr[row + 1] = len(indices)
        return sparse.csr_matrix(
            (
                np.asarray(data, dtype=np.float64),
                np.asarray(indices, dtype=np.int32),
                indptr,
            ),
            shape=(n_rows, n_cols),
        )

    def get_feature_names_out(self, input_features=None):
        names: list[str | None] = [None] * len(self.vocabulary_)
        for key, index in self.vocabulary_.items():
            names[index] = key
        return np.asarray(names, dtype=object)


# --------------------------------------------------------------------------
# Fabricas
# --------------------------------------------------------------------------


def assert_unfitted(estimator) -> None:
    leaked = [
        name
        for name in ("vocabulary_", "mean_", "statistics_", "classes_", "coef_")
        if hasattr(estimator, name)
    ]
    if leaked:
        raise RuntimeError(
            f"{estimator.__class__.__name__} llego ajustado ({', '.join(leaked)}). "
            "Las fabricas tienen que entregar estimadores sin fit."
        )
    steps = getattr(estimator, "named_steps", None)
    if steps is not None:
        for step in steps.values():
            assert_unfitted(step)
    if estimator.__class__.__name__ == "ColumnTransformer":
        for _, nested, _ in estimator.transformers:
            assert_unfitted(nested)


def make_aggregate_steps() -> list:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("binary", binary_pipeline, BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return [
        ("chord_features", ChordFeaturesTransformer()),
        ("preprocess", preprocess),
    ]


def make_aggregate_preprocessor() -> Pipeline:
    return Pipeline(steps=make_aggregate_steps())


def make_ngram_transformer(name: str) -> ChordNgramTransformer:
    """`name` es '<alfabeto>_<orden>', por ejemplo 'degree_4' o 'absolute_3'."""
    alphabet, order = name.split("_", 1)
    return ChordNgramTransformer(
        n=int(order),
        alphabet=alphabet,
        min_df=NGRAM_MIN_DF,
        max_features=NGRAM_MAX_FEATURES,
    )


def make_sgd(params: dict) -> SGDClassifier:
    kwargs = {
        "loss": "log_loss",
        "penalty": params["penalty"],
        "alpha": params["alpha"],
        "class_weight": params["class_weight"],
        "random_state": SEED,
        "max_iter": 1000,
        "tol": 1e-3,
    }
    if params["penalty"] == "elasticnet":
        kwargs["l1_ratio"] = params["l1_ratio"]
    return SGDClassifier(**kwargs)


def build_estimator(kind: str, params: dict):
    if kind == "dummy":
        return DummyClassifier(strategy="most_frequent", random_state=SEED)
    if kind == "complement_nb":
        return ComplementNB(alpha=params["alpha"])
    if kind == "sgd":
        return make_sgd(params)
    if kind == "linear_svc":
        return LinearSVC(
            C=params.get("C", 1.0),
            class_weight="balanced",
            dual=False,
            random_state=SEED,
            max_iter=100,
        )
    raise ValueError(f"Estimador desconocido: {kind}")


def make_pipeline(representation: str, estimator) -> Pipeline:
    """`representation` es 'aggregate' o un nombre de n-grama ('degree_4')."""
    if representation == "aggregate":
        return Pipeline(steps=make_aggregate_steps() + [("classifier", estimator)])
    return Pipeline(
        steps=[
            ("ngrams", make_ngram_transformer(representation)),
            ("classifier", estimator),
        ]
    )
