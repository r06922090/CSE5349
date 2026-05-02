from __future__ import annotations

import csv as _csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator


@dataclass
class EffectRow:
    crate: str
    fn_decl: str
    callee: str
    effect: str
    dir: str
    file: str
    line: int
    col: int
    human: str
    gpt: str
    source_csv: str = ""
    row_index: int = 0


def split_escaped(line: str) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "\\" and i + 1 < len(line) and line[i + 1] == ",":
            buf.append(",")
            i += 2
            continue
        if ch == ",":
            out.append("".join(buf).strip())
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    out.append("".join(buf).strip())
    return out


def parse_row(
    fields: list[str],
) -> tuple[str, str, str, str, str, str, int, int, str, str]:
    """Normalize 9- or 10-field row into the 10-field shape."""
    if len(fields) == 10:
        crate, fn_decl, callee, effect, dir_, file, line, col, human, gpt = fields
    elif len(fields) == 9:
        # Missing `file` column, row layout: crate, fn_decl, callee, effect, dir, line, col, human, gpt
        crate, fn_decl, callee, effect, dir_, line, col, human, gpt = fields
        file = ""
    else:
        raise ValueError(f"Unexpected field count {len(fields)}: {fields}")
    return crate, fn_decl, callee, effect, dir_, file, int(line), int(col), human, gpt


def iter_rows(csv_path: Path) -> Iterator[EffectRow]:
    text = csv_path.read_text()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return
    # Skip header.
    for idx, raw in enumerate(lines[1:], start=1):
        fields = split_escaped(raw)
        try:
            (crate, fn_decl, callee, effect, dir_, file, line, col, human, gpt) = (
                parse_row(fields)
            )
        except ValueError as e:
            raise ValueError(f"{csv_path}:{idx + 1}: {e}") from None
        yield EffectRow(
            crate=crate,
            fn_decl=fn_decl,
            callee=callee,
            effect=effect,
            dir=dir_,
            file=file,
            line=line,
            col=col,
            human=human,
            gpt=gpt,
            source_csv=str(csv_path),
            row_index=idx,
        )


def iter_dataset(
    dataset_dir: Path, crates: Iterable[str] | None = None
) -> Iterator[EffectRow]:
    crates_set = set(crates) if crates else None
    for csv_path in sorted(dataset_dir.glob("*.csv")):
        if crates_set is not None and csv_path.stem not in crates_set:
            continue
        yield from iter_rows(csv_path)


def write_predictions(
    rows: list[tuple[EffectRow, dict]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[tuple[EffectRow, dict]]] = {}
    for row, pred in rows:
        grouped.setdefault(row.crate, []).append((row, pred))

    fieldnames = [
        "crate",
        "fn_decl",
        "callee",
        "effect",
        "dir",
        "file",
        "line",
        "col",
        "human",
        "gpt",
        "agent_verdict",
        "agent_rationale",
        "agent_trace_path",
    ]
    for crate, items in grouped.items():
        out_csv = out_dir / f"{crate}.csv"
        with out_csv.open("w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row, pred in items:
                d = asdict(row)
                d.pop("source_csv", None)
                d.pop("row_index", None)
                d["agent_verdict"] = pred.get("verdict", "")
                d["agent_rationale"] = pred.get("rationale", "")
                d["agent_trace_path"] = pred.get("trace_path", "")
                w.writerow(d)
