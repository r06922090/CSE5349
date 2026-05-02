#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tree_sitter import Language, Parser
import tree_sitter_rust


@dataclass
class FunctionRecord:
    path: str
    code: str
    file: str
    line_start: int
    line_end: int


def build_parser() -> Parser:
    rust_lang = Language(tree_sitter_rust.language())
    parser = Parser()
    parser.language = rust_lang
    return parser


def decode_node_text(source_bytes: bytes, node) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode(
        "utf-8", errors="replace"
    )


def node_field_text(source_bytes: bytes, node, field_name: str) -> str | None:
    child = node.child_by_field_name(field_name)
    if child is None:
        return None
    return decode_node_text(source_bytes, child).strip()


def normalize_type_name(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = " ".join(raw.split())
    s = s.replace("& mut ", "&mut ")
    return s


# Convert a Rust file path into its module path relative to src_root.
def rust_file_module_path(src_root: Path, rust_file: Path) -> list[str]:
    rel = rust_file.relative_to(src_root)
    parts = list(rel.parts)

    if not parts:
        return []

    if parts[-1] == "lib.rs" or parts[-1] == "main.rs":
        parts = parts[:-1]
    elif parts[-1] == "mod.rs":
        parts = parts[:-1]
    else:
        parts[-1] = Path(parts[-1]).stem

    return [p for p in parts if p]


def iter_rust_files(src_root: Path) -> Iterable[Path]:
    for path in src_root.rglob("*.rs"):
        if path.is_file():
            yield path


def create_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS functions (
            path TEXT PRIMARY KEY,
            code TEXT NOT NULL,
            file TEXT NOT NULL,
            line_start INTEGER NOT NULL,
            line_end INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_functions_file
        ON functions(file)
        """
    )
    conn.commit()


def insert_record(conn: sqlite3.Connection, rec: FunctionRecord) -> None:
    conn.execute(
        """
        INSERT INTO functions (path, code, file, line_start, line_end)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            code = excluded.code,
            file = excluded.file,
            line_start = excluded.line_start,
            line_end = excluded.line_end
        """,
        (rec.path, rec.code, rec.file, rec.line_start, rec.line_end),
    )


def collect_functions_from_node(
    node,
    source_bytes: bytes,
    file_path: Path,
    module_stack: list[str],
    impl_stack: list[str],
    out: list[FunctionRecord],
) -> None:
    node_type = node.type

    if node_type == "mod_item":
        mod_name = node_field_text(source_bytes, node, "name")
        next_module_stack = module_stack
        if mod_name:
            next_module_stack = module_stack + [mod_name]

        for child in node.named_children:
            collect_functions_from_node(
                child,
                source_bytes,
                file_path,
                next_module_stack,
                impl_stack,
                out,
            )
        return

    if node_type == "impl_item":
        type_name = normalize_type_name(node_field_text(source_bytes, node, "type"))
        trait_name = normalize_type_name(node_field_text(source_bytes, node, "trait"))

        next_impl_stack = impl_stack.copy()
        if type_name:
            next_impl_stack.append(type_name)
        if trait_name:
            next_impl_stack.append(trait_name)

        for child in node.named_children:
            collect_functions_from_node(
                child,
                source_bytes,
                file_path,
                module_stack,
                next_impl_stack,
                out,
            )
        return

    if node_type == "function_item":
        fn_name = node_field_text(source_bytes, node, "name")
        if fn_name:
            full_parts = module_stack + impl_stack + [fn_name]
            full_path = "::".join(full_parts)
            code = decode_node_text(source_bytes, node)
            out.append(
                FunctionRecord(
                    path=full_path,
                    code=code,
                    file=str(file_path),
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                )
            )

    for child in node.named_children:
        collect_functions_from_node(
            child,
            source_bytes,
            file_path,
            module_stack,
            impl_stack,
            out,
        )


def parse_rust_file(
    parser: Parser, src_root: Path, rust_file: Path
) -> list[FunctionRecord]:
    source_bytes = rust_file.read_bytes()
    tree = parser.parse(source_bytes)
    root = tree.root_node

    base_module_path = rust_file_module_path(src_root, rust_file)
    records: list[FunctionRecord] = []

    collect_functions_from_node(
        node=root,
        source_bytes=source_bytes,
        file_path=rust_file,
        module_stack=base_module_path,
        impl_stack=[],
        out=records,
    )
    return records


def build_database(src_root: Path, db_path: Path) -> None:
    parser = build_parser()
    conn = sqlite3.connect(str(db_path))
    try:
        create_schema(conn)

        total = 0
        for rust_file in iter_rust_files(src_root):
            records = parse_rust_file(parser, src_root, rust_file)
            for rec in records:
                insert_record(conn, rec)
            total += len(records)

        conn.commit()
        print(f"Indexed {total} functions into {db_path}")
    finally:
        conn.close()


def main() -> None:
    argp = argparse.ArgumentParser(
        description="Parse Rust files with Tree-sitter and store functions in SQLite."
    )
    argp.add_argument(
        "src_root",
        type=Path,
        help="Rust source root, usually something like ./src",
    )
    argp.add_argument(
        "db_path",
        type=Path,
        help="Output SQLite database path, for example functions.db",
    )
    args = argp.parse_args()

    src_root = args.src_root.resolve()
    db_path = args.db_path.resolve()

    if not src_root.exists() or not src_root.is_dir():
        raise SystemExit(
            f"Source root does not exist or is not a directory: {src_root}"
        )

    build_database(src_root, db_path)


if __name__ == "__main__":
    main()
