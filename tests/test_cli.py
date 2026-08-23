"""Tests for the AVT command-line interface."""

from __future__ import annotations

import pytest

from avt.cli import build_parser, main


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "avt 0.1.0" in capsys.readouterr().out


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "usage:" in out
    assert "select-tasks" in out
    assert "generate" in out


def test_unknown_command_fails(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["not-a-command"])
    assert exc.value.code == 2


def test_select_tasks_requires_config(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["select-tasks"])
    assert exc.value.code == 2


def test_rank_selector_choices() -> None:
    parser = build_parser()
    assert parser.parse_args(["rank", "--config", "x.yaml"]).selector == "continuous"
    assert (
        parser.parse_args(["rank", "--config", "x.yaml", "--selector", "discrete"]).selector
        == "discrete"
    )
