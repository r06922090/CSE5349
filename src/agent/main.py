from __future__ import annotations

import argparse
import sys

from .agent import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.agent.main")
    parser.add_argument("--provider", choices=["anthropic", "openai"], required=True)
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit total rows audited"
    )
    parser.add_argument(
        "--crates",
        type=str,
        default=None,
        help="Comma-separated crate names to include (matches CSV stem)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=8, help="Per-row tool-call budget"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output (one line per row); default is verbose streaming",
    )

    args = parser.parse_args(argv)
    crates = [c.strip() for c in args.crates.split(",")] if args.crates else None
    run(
        provider_name=args.provider,
        crates=crates,
        limit=args.limit,
        max_steps=args.max_steps,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
