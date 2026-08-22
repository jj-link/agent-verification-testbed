"""Command-line interface for the Agent Verification Testbed."""

from __future__ import annotations

import argparse

__all__ = ["main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="avt",
        description="Agent Verification Testbed",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )
    parser.add_argument(
        "command",
        nargs="?",
        help="command to run, e.g. 'doctor'",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command != "doctor":
        parser.error(f"unknown command: {args.command}")
    from avt.doctor import format_results, run_doctor

    results = run_doctor()
    print(format_results(results))
    return 0 if all(r.ok for r in results) else 1
