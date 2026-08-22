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
        default="doctor",
        help="command slot for forthcoming experiment stages",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        print("avt: doctor requires configuration from later stages")
    else:
        parser = build_parser()
        parser.error(f"unknown command: {args.command}")
    return 0
