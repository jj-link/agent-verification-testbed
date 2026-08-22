"""Tests for the AVT command-line interface."""

from __future__ import annotations

import pytest

from avt.cli import main


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "avt 0.1.0" in capsys.readouterr().out


def test_default_command_is_doctor(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "avt: doctor" in capsys.readouterr().out


def test_unknown_command_fails(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["not-a-command"])
    assert exc.value.code == 2
    assert "unknown command" in capsys.readouterr().err
