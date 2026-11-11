"""Carga de datos y corte desarrollo / holdout.

Un solo lugar define que filas son de desarrollo ("dev") y cuales del holdout.
El EDA, el notebook 02 y el notebook 03 usan estas funciones, asi ninguno
mira el holdout por su cuenta.

Reglas del corte:
  1. Holdout: todas las filas de los artistas listados en
     data/holdout_artists.json (sorteados una vez por data/make_holdout.py).
  2. Dev: todo lo demas, menos las filas cuyo string `chords` aparece tambien
     en el holdout (progresiones repetidas entre artistas distintos). Esas
     copias se descartan de dev para que el modelo no vea en entrenamiento
     la misma cancion que luego se evalua.

Solo se comparan strings `chords` exactos. Una cancion transpuesta o con otra
estructura no se detecta.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ARTISTS_FILE = "holdout_artists.json"
MANIFEST_FILE = "holdout_manifest.json"


def load_shards(data_dir: Path, verbose: bool = True) -> pd.DataFrame:
    paths = sorted(Path(data_dir).glob("chordonomicon_v2_part_*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No hay chordonomicon_v2_part_*.csv en {Path(data_dir).resolve()}."
        )
    frames: list[pd.DataFrame] = []
    header: list[str] | None = None
    for path in paths:
        # Un solo pase de tipos. Con el lector por trozos, un bloque de nulos
        # se infiere como numero y el siguiente como texto.
        part = pd.read_csv(path, low_memory=False)
        columns = list(part.columns)
        if header is None:
            header = columns
        elif columns != header:
            raise ValueError(
                f"{path.name} no tiene el mismo encabezado que {paths[0].name}: {columns}"
            )
        frames.append(part)
        if verbose:
            print(f"{path.name}: {len(part):,} filas")
    return pd.concat(frames, ignore_index=True)


def load_holdout_spec(data_dir: Path) -> tuple[list[str], dict]:
    """Lista de artistas del holdout y manifiesto, con el hash verificado."""
    data_dir = Path(data_dir)
    artists = json.loads((data_dir / ARTISTS_FILE).read_text(encoding="utf8"))
    manifest = json.loads((data_dir / MANIFEST_FILE).read_text(encoding="utf8"))
    digest = hashlib.sha256(json.dumps(artists).encode("utf8")).hexdigest()
    if digest != manifest["holdout_artists_sha256"]:
        raise ValueError(f"{ARTISTS_FILE} no coincide con el hash del manifiesto.")
    return artists, manifest


def holdout_mask(df: pd.DataFrame, holdout_artists: list[str]) -> np.ndarray:
    """True en las filas de los artistas del holdout."""
    return df["artist_id"].astype(str).isin(set(holdout_artists)).to_numpy()


def dev_mask(df: pd.DataFrame, holdout_artists: list[str]) -> np.ndarray:
    """True en las filas de desarrollo: ni del holdout ni copia exacta de el."""
    in_holdout = holdout_mask(df, holdout_artists)
    holdout_chords = set(df.loc[in_holdout, "chords"].dropna())
    duplicate = df["chords"].isin(holdout_chords).to_numpy() & ~in_holdout
    return ~in_holdout & ~duplicate


def load_dev(data_dir: Path, verbose: bool = True) -> pd.DataFrame:
    """Todas las filas de desarrollo (con y sin etiqueta). Sin holdout."""
    raw = load_shards(data_dir, verbose=verbose)
    artists, manifest = load_holdout_spec(data_dir)
    keep = dev_mask(raw, artists)
    dev = raw.loc[keep].reset_index(drop=True)
    if len(dev) != manifest["n_rows_dev_all"]:
        raise ValueError(
            f"dev tiene {len(dev):,} filas y el manifiesto dice {manifest['n_rows_dev_all']:,}."
        )
    return dev
