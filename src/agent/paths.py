from __future__ import annotations

from functools import lru_cache
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CRATES_ROOT = REPO_ROOT / "data" / "crates"
DATASET_ROOT = REPO_ROOT / "data" / "dataset"
PREDICTIONS_ROOT = REPO_ROOT / "data" / "dataset_predictions"


@lru_cache(maxsize=None)
def resolve_crate_root(crate: str) -> Path:
    matches = sorted(CRATES_ROOT.glob(f"{crate}-*"))
    matches = [m for m in matches if m.is_dir()]
    if not matches:
        raise FileNotFoundError(f"No crate dir found for {crate!r} under {CRATES_ROOT}")
    if len(matches) > 1:
        # Prefer the latest version.
        matches.sort()
    return matches[-1]


def crate_src_root(crate: str) -> Path:
    return resolve_crate_root(crate) / "src"


def crate_db_path(crate: str) -> Path:
    return resolve_crate_root(crate) / "functions.db"
