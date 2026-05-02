from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from .csv_io import EffectRow, iter_dataset, write_predictions
from .db import ensure_db
from .paths import DATASET_ROOT, PREDICTIONS_ROOT
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .providers import Provider, get_provider
from .tools import build_tools


def truncate(s: str, n: int) -> str:
    s = s.strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "..."


def summarize_tool_input(name: str, args: dict) -> str:
    if name == "lookup_function":
        return f"path={args.get('path', '')!r}"
    if name == "read_around":
        return (
            f"file={args.get('file', '')!r} line={args.get('line')} "
            f"-{args.get('before', 20)}/+{args.get('after', 20)}"
        )
    if name == "search_functions":
        return f"query={args.get('query', '')!r} limit={args.get('limit', 10)}"
    if name == "read_file":
        return f"file={args.get('file', '')!r} max_lines={args.get('max_lines', 400)}"
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


def summarize_tool_result(name: str, result: dict) -> str:
    if not isinstance(result, dict):
        return truncate(str(result), 80)
    if "error" in result:
        return f"error: {truncate(result['error'], 100)}"
    if name == "lookup_function":
        if not result.get("found"):
            return "not found (likely external/std)"
        return f"found {result.get('file')}:{result.get('line_start')}-{result.get('line_end')}"
    if name == "read_around":
        lines = result.get("lines") or result.get("content") or ""
        n = lines.count("\n") + 1 if isinstance(lines, str) and lines else 0
        if "error" in result:
            return truncate(str(result.get("error")), 100)
        return f"{n} lines"
    if name == "search_functions":
        hits = result.get("results", [])
        return f"{len(hits)} hits"
    if name == "read_file":
        content = result.get("content") or result.get("lines") or ""
        n = content.count("\n") + 1 if isinstance(content, str) and content else 0
        return f"{n} lines"
    return truncate(json.dumps(result, default=str), 100)


def make_printer(verbose: bool, idx: int, total: int, row: EffectRow):
    state = {"step": 0, "tool_count": 0}

    if not verbose:
        # quiet mode: only header + footer
        def on_event(_e: dict) -> None:
            pass

        return on_event, state

    # verbose header
    print()
    bar = f"[{idx}/{total}] {row.crate}"
    print(f"{bar} {row.callee}  {f'(effect: {row.effect})'}")
    print(f"{'fn'} {row.fn_decl}")
    site = f"{row.file or '?'}:{row.line}" if row.line else (row.file or "?")
    print(f"{'site'} {site}  {f'dir={row.dir}'}")

    def on_event(e: dict) -> None:
        kind = e.get("step")
        if kind == "text":
            text = truncate(e.get("text", ""), 240)
            if text:
                print(f"{'think'} {text}")
        elif kind == "tool_start":
            state["tool_count"] += 1
            name = e.get("name", "?")
            args = e.get("input", {}) or {}
            print(f"{'tool'} {name}({summarize_tool_input(name, args)})")
        elif kind == "tool_end":
            name = e.get("name", "?")
            print(f"  {'result'} {summarize_tool_result(name, e.get('result', {}))}")
        elif kind == "final":
            v = e.get("verdict", {}) or {}
            verdict = v.get("verdict", "?")
            rationale = truncate(v.get("rationale", ""), 320)
            print(f"{'final'} verdict={verdict}")
            if rationale:
                print(f"{'rationale'} {rationale}")
        elif kind == "budget_exhausted":
            print(f"{'budget exhausted'} after {e.get('steps')} steps")

    return on_event, state


def print_footer(
    verdict: str,
    rationale: str,
    steps: int,
    elapsed: float,
    verbose: bool,
    idx: int = 0,
    total: int = 0,
    row: EffectRow | None = None,
) -> None:
    if verbose:
        print(f"{verdict} {f'({steps} steps, {elapsed:.1f}s)'}")
    else:
        # mirrors prior format in quiet mode
        print(
            f"[{idx}/{total}] {row.crate} :: {row.callee} ({row.effect})",
            flush=True,
        )
        print(f"{verdict}", flush=True)


def audit_row(
    provider: Provider,
    row: EffectRow,
    traces_dir: Path,
    max_steps: int = 8,
    verbose: bool = True,
    idx: int = 0,
    total: int = 0,
) -> dict:
    db_path = ensure_db(row.crate)
    tools = build_tools(row.crate, db_path)
    user = build_user_prompt(row)

    on_event, _state = make_printer(verbose, idx, total, row)
    t0 = time.monotonic()
    try:
        result = provider.run(
            SYSTEM_PROMPT, user, tools, max_steps=max_steps, on_event=on_event
        )
    except Exception as e:
        elapsed = time.monotonic() - t0
        rationale = f"[provider-error] {type(e).__name__}: {e}"
        if verbose:
            print(f"{'provider-error'} {type(e).__name__}: {e}")
        print_footer("error", rationale, 0, elapsed, verbose, idx, total, row)
        return {
            "verdict": "error",
            "rationale": rationale,
            "trace_path": "",
            "error": traceback.format_exc(),
        }

    elapsed = time.monotonic() - t0
    trace_file = traces_dir / f"{row.crate}__{row.row_index:03d}.jsonl"
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    with trace_file.open("w") as f:
        for step in result.get("trace", []):
            f.write(json.dumps(step, ensure_ascii=False, default=str))
            f.write("\n")

    verdict = result.get("verdict", "error")
    rationale = result.get("rationale", "")
    steps = result.get("steps", 0)
    print_footer(verdict, rationale, steps, elapsed, verbose, idx, total, row)

    return {
        "verdict": verdict,
        "rationale": rationale,
        "trace_path": str(trace_file.relative_to(PREDICTIONS_ROOT.parent)),
    }


def run(
    provider_name: str,
    crates: list[str] | None = None,
    limit: int | None = None,
    max_steps: int = 8,
    out_root: Path | None = None,
    verbose: bool = True,
) -> Path:
    provider = get_provider(provider_name)
    out_root = out_root or PREDICTIONS_ROOT / provider.name
    traces_dir = out_root / "traces"
    out_root.mkdir(parents=True, exist_ok=True)

    rows = list(iter_dataset(DATASET_ROOT, crates=crates))
    if limit is not None:
        rows = rows[:limit]

    model = getattr(provider, "model", "?")
    print(
        f"[agent] {provider.name} ({model}): auditing {len(rows)} rows",
        flush=True,
    )
    results: list[tuple[EffectRow, dict]] = []
    safe_n = unsafe_n = error_n = 0
    t_start = time.monotonic()
    for i, row in enumerate(rows, start=1):
        pred = audit_row(
            provider,
            row,
            traces_dir,
            max_steps=max_steps,
            verbose=verbose,
            idx=i,
            total=len(rows),
        )
        results.append((row, pred))
        v = pred["verdict"]
        if v == "safe":
            safe_n += 1
        elif v == "unsafe":
            unsafe_n += 1
        else:
            error_n += 1

    write_predictions(results, out_root)
    total_elapsed = time.monotonic() - t_start
    print()
    print(
        f"[agent] done in {total_elapsed:.1f}s -- "
        f"{f'{safe_n} safe'} / {f'{unsafe_n} unsafe'} / {f'{error_n} error'}"
    )
    print(f"[agent] wrote predictions to {out_root}", flush=True)
    return out_root
