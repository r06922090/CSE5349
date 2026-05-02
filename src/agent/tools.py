from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import db


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON schema for the tool's input
    runner: Callable[[dict], dict]


def build_tools(crate: str, db_path: Path) -> list[Tool]:

    def lookup_function(args: dict) -> dict:
        path = args.get("path", "")
        hit = db.lookup_by_path(db_path, path, crate=crate)
        if hit is None:
            return {"found": False, "query": path}
        return {
            "found": True,
            "path": hit.path,
            "file": hit.file,
            "line_start": hit.line_start,
            "line_end": hit.line_end,
            "code": hit.code,
        }

    def read_around(args: dict) -> dict:
        return db.read_around(
            crate,
            file=args["file"],
            line=int(args["line"]),
            before=int(args.get("before", 20)),
            after=int(args.get("after", 20)),
        )

    def search_functions(args: dict) -> dict:
        hits = db.search_by_path(
            db_path, args["query"], limit=int(args.get("limit", 10))
        )
        return {
            "results": [
                {
                    "path": h.path,
                    "file": h.file,
                    "line_start": h.line_start,
                    "line_end": h.line_end,
                    "preview": h.code.splitlines()[0] if h.code else "",
                }
                for h in hits
            ]
        }

    def read_file(args: dict) -> dict:
        return db.read_file(
            crate,
            file=args["file"],
            max_lines=int(args.get("max_lines", 400)),
        )

    return [
        Tool(
            name="lookup_function",
            description=(
                "Look up a Rust function by its module path (e.g. 'abomonated::Abomonated::new')."
                "The crate prefix is optional. Returns the full function code, file, and line range."
                "Returns {found: false} if the function is defined outside this crate (e.g. std, libc, or a dependency)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Function module path, with or without crate prefix.",
                    },
                },
                "required": ["path"],
            },
            runner=lookup_function,
        ),
        Tool(
            name="read_around",
            description=(
                "Read source lines around a specific (file, line) location."
                "Use this to inspect the actual callsite of an effect."
                "The file is relative to the crate root (e.g. 'src/lib.rs') or a bare filename which will be searched under src/."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "line": {"type": "integer"},
                    "before": {"type": "integer", "default": 20},
                    "after": {"type": "integer", "default": 20},
                },
                "required": ["file", "line"],
            },
            runner=read_around,
        ),
        Tool(
            name="search_functions",
            description=(
                "Substring search over indexed function paths in this crate."
                "Use this when you need to find callers or related helpers and don't have the exact path."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
            runner=search_functions,
        ),
        Tool(
            name="read_file",
            description=(
                "Read up to `max_lines` lines (default 400) of a source file."
                "Prefer `lookup_function` and `read_around` first; only fall back to this when targeted reads are insufficient."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "max_lines": {"type": "integer", "default": 400},
                },
                "required": ["file"],
            },
            runner=read_file,
        ),
    ]


def run_tool(tools: list[Tool], name: str, args: dict) -> Any:
    for t in tools:
        if t.name == name:
            try:
                return t.runner(args)
            except Exception as e:
                return {"error": f"{type(e).__name__}: {e}"}
    return {"error": f"unknown tool: {name}"}
