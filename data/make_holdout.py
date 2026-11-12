"""Crea el holdout final agrupado por artista y su manifiesto.

Se ejecuta una sola vez. No imprime ni guarda etiquetas del holdout: solo
necesita saber que main_genre no es nulo para definir el universo supervisado.

Salida (en data/):
  holdout_artists.json   lista ordenada de artist_id del holdout (como texto)
  holdout_manifest.json  semilla, tamanos, conteos de duplicados, sha256 y versiones

Uso, desde la raiz del repo:
  python data/make_holdout.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupShuffleSplit

DATA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA_DIR.parent / "src"))
import chorddata  # noqa: E402

SEED = 2026
TEST_SIZE = 0.20

ARTISTS_FILE = DATA_DIR / chorddata.ARTISTS_FILE
MANIFEST_FILE = DATA_DIR / chorddata.MANIFEST_FILE


def main() -> None:
    if ARTISTS_FILE.exists() or MANIFEST_FILE.exists():
        raise SystemExit(
            "El holdout ya existe. No se regenera: hacerlo invalida cualquier "
            "evaluacion previa. Borra los dos JSON a mano solo si sabes por que."
        )

    raw = pd.concat(
        [
            pd.read_csv(path, usecols=["id", "chords", "artist_id", "main_genre"], low_memory=False)
            for path in sorted(DATA_DIR.glob("chordonomicon_v2_part_*.csv"))
        ],
        ignore_index=True,
    )
    if raw.empty:
        raise SystemExit(f"No hay chordonomicon_v2_part_*.csv en {DATA_DIR}.")

    labeled = raw["main_genre"].notna().to_numpy()
    supervised = raw.loc[labeled, ["id", "artist_id"]]
    if supervised["artist_id"].isna().any():
        raise SystemExit("Hay filas con main_genre y sin artist_id.")

    # Mismo sorteo que antes: el holdout sale solo de artistas con etiqueta.
    artist = supervised["artist_id"].astype(str).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    dev_idx, hold_idx = next(splitter.split(supervised, groups=artist))

    holdout_artists = sorted(set(artist[hold_idx]))
    dev_artists = set(artist[dev_idx])
    if dev_artists.intersection(holdout_artists):
        raise SystemExit("Un artista quedo en dev y en holdout.")

    # Duplicados exactos de `chords` entre dev y holdout, sobre todas las filas.
    in_holdout = chorddata.holdout_mask(raw, holdout_artists)
    keep = chorddata.dev_mask(raw, holdout_artists)
    n_dropped = int((~in_holdout & ~keep).sum())
    n_dropped_labeled = int((~in_holdout & ~keep & labeled).sum())

    digest = hashlib.sha256(json.dumps(holdout_artists).encode("utf8")).hexdigest()
    manifest = {
        "seed": SEED,
        "test_size": TEST_SIZE,
        "universe": "filas con main_genre no nulo",
        "n_rows_all": int(len(raw)),
        "n_rows_supervised": int(labeled.sum()),
        "n_artists_supervised": int(len(set(artist))),
        "n_artists_dev": int(len(dev_artists)),
        "n_artists_holdout": int(len(holdout_artists)),
        "n_rows_holdout": int(in_holdout.sum()),
        "duplicate_rule": "filas fuera del holdout cuyo string chords exacto aparece en el holdout",
        "n_rows_dropped_duplicates_all": n_dropped,
        "n_rows_dropped_duplicates_labeled": n_dropped_labeled,
        "n_rows_dev_all": int(keep.sum()),
        "n_rows_dev_labeled": int((keep & labeled).sum()),
        "holdout_artists_sha256": digest,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }

    ARTISTS_FILE.write_text(json.dumps(holdout_artists), encoding="utf8")
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print(json.dumps(manifest, indent=2))
    print("No se imprimen etiquetas del holdout.", file=sys.stderr)


if __name__ == "__main__":
    main()
