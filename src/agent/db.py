from __future__ import annotations

import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .paths import REPO_ROOT, crate_db_path, crate_src_root, resolve_crate_root


TREE_SITTER_SCRIPT = REPO_ROOT / "src" / "tree-sitter-test.py"


@dataclass
class FunctionHit:
    path: str
    code: str
    file: str
    line_start: int
    line_end: int


def ensure_db(crate: str) -> Path:
    db = crate_db_path(crate)
    if db.exists() and db.stat().st_size > 0:
        return db
    src = crate_src_root(crate)
    if not src.is_dir():
        raise FileNotFoundError(f"No src dir for {crate}: {src}")
    cmd = [sys.executable, str(TREE_SITTER_SCRIPT), str(src), str(db)]
    subprocess.run(cmd, check=True)
    return db


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def normalize_query(fn_path: str, crate: str | None) -> list[str]:
    raw = fn_path.strip()
    candidates: list[str] = [raw]

    if crate and raw.startswith(crate + "::"):
        stripped = raw[len(crate) + 2 :]
        candidates.append(stripped)
    else:
        stripped = raw

    # Rewrite `<X as Trait>::m` to `X::Trait::m`
    if "<" in stripped and " as " in stripped:
        rewritten = stripped
        while True:
            i = rewritten.find("<")
            j = rewritten.find(" as ", i)
            k = rewritten.find(">::", j) if j != -1 else -1
            if i == -1 or j == -1 or k == -1:
                break
            ty = rewritten[i + 1 : j]
            tr = rewritten[j + 4 : k]
            # k points at '>'; drop just '>' so the '::' that follows is preserved
            rewritten = rewritten[:i] + ty + "::" + tr + rewritten[k + 1 :]
        candidates.append(rewritten)

    # Deduplicate and preserve order
    seen, out = set(), []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def lookup_by_path(
    db_path: Path, fn_path: str, crate: str | None = None
) -> FunctionHit | None:
    queries = normalize_query(fn_path, crate)
    with connect(db_path) as conn:
        for q in queries:
            cur = conn.execute(
                "SELECT path, code, file, line_start, line_end FROM functions WHERE path = ?",
                (q,),
            )
            row = cur.fetchone()
            if row is not None:
                return FunctionHit(**dict(row))
        # Fallback: match by the function name plus immediate parent type, tolerating generic parameters between segments.
        for q in queries:
            segs = q.split("::")
            if not segs:
                continue
            fn_name = segs[-1]
            parent = segs[-2] if len(segs) >= 2 else ""
            if parent:
                pattern = f"%{parent}%::{fn_name}"
            else:
                pattern = f"%::{fn_name}"
            cur = conn.execute(
                "SELECT path, code, file, line_start, line_end FROM functions "
                "WHERE path LIKE ? OR path = ? ORDER BY length(path) ASC LIMIT 1",
                (pattern, fn_name),
            )
            row = cur.fetchone()
            if row is not None:
                return FunctionHit(**dict(row))
    return None


def search_by_path(db_path: Path, needle: str, limit: int = 10) -> list[FunctionHit]:
    needle = needle.strip()
    if not needle:
        return []
    with connect(db_path) as conn:
        cur = conn.execute(
            "SELECT path, code, file, line_start, line_end FROM functions "
            "WHERE path LIKE ? ORDER BY length(path) ASC LIMIT ?",
            (f"%{needle}%", limit),
        )
        rows = cur.fetchall()
    return [FunctionHit(**dict(r)) for r in rows]


def read_around(
    crate: str, file: str, line: int, before: int = 20, after: int = 20
) -> dict:
    root = resolve_crate_root(crate)
    target = resolve_file(root, file)
    text_lines = target.read_text(errors="replace").splitlines()
    n = len(text_lines)
    lo = max(1, line - before)
    hi = min(n, line + after)
    out = []
    for i in range(lo, hi + 1):
        out.append(f"{i:>5}: {text_lines[i - 1]}")
    return {
        "file": str(target.relative_to(root)),
        "line_start": lo,
        "line_end": hi,
        "content": "\n".join(out),
    }


def read_file(crate: str, file: str, max_lines: int = 400) -> dict:
    root = resolve_crate_root(crate)
    target = resolve_file(root, file)
    text_lines = target.read_text(errors="replace").splitlines()
    truncated = len(text_lines) > max_lines
    shown = text_lines[:max_lines]
    out = [f"{i:>5}: {ln}" for i, ln in enumerate(shown, start=1)]
    return {
        "file": str(target.relative_to(root)),
        "line_start": 1,
        "line_end": len(shown),
        "total_lines": len(text_lines),
        "truncated": truncated,
        "content": "\n".join(out),
    }


def resolve_file(crate_root: Path, file: str) -> Path:
    p = Path(file)
    if p.is_absolute():
        candidate = p
    else:
        # Try to find the file in the crate root and src dir in first.
        candidates = [
            crate_root / p,
            crate_root / "src" / p,
        ]
        # Search in src dir to find the file
        if len(p.parts) == 1:
            for found in (crate_root / "src").rglob(p.name):
                if found.is_file():
                    candidates.append(found)
                    break
        candidate = next((c for c in candidates if c.exists()), candidates[0])

    candidate = candidate.resolve()
    crate_root_resolved = crate_root.resolve()
    try:
        candidate.relative_to(crate_root_resolved)
    except ValueError:
        raise PermissionError(f"File outside crate root: {candidate}")
    if not candidate.is_file():
        raise FileNotFoundError(f"Not a file: {file}")
    return candidate
