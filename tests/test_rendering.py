"""Tests for the deterministic, verifier-safe trajectory renderer."""

from __future__ import annotations

import json
from pathlib import Path

from avt.rendering import estimate_tokens, render_trajectory


def _atif(steps: list[dict[str, object]]) -> dict[str, object]:
    return {"schema_version": "ATIF-v1.6", "steps": steps}


def _user(msg: str) -> dict[str, object]:
    return {"step_id": 1, "source": "user", "message": msg}


def _agent(
    text: str = "",
    calls: list[tuple[str, dict[str, object]]] | None = None,
    outputs: list[str] | None = None,
) -> dict[str, object]:
    d: dict[str, object] = {"step_id": 2, "source": "agent", "message": text}
    if calls:
        d["tool_calls"] = [
            {"tool_call_id": f"call_{i}", "function_name": name, "arguments": args}
            for i, (name, args) in enumerate(calls)
        ]
    if outputs:
        d["observation"] = {
            "results": [
                {"source_call_id": f"call_{i}", "content": content}
                for i, content in enumerate(outputs)
            ]
        }
    return d


def _write(tmp_path: Path, atif: dict[str, object], name: str = "trajectory.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(atif), encoding="utf-8")
    return p


def test_render_keeps_instruction_and_actions_in_order(tmp_path: Path) -> None:
    atif = _atif(
        [
            _user("PUBLIC INSTRUCTION LINE"),
            _user("CONTEXT LINE 2"),
            _agent(text="assistant reasoning"),
            _agent(calls=[("shell", {"cmd": "ls -la"})], outputs=["file listing output"]),
        ]
    )
    r = render_trajectory("C1", "task_x", _write(tmp_path, atif), body_budget_tokens=10**6)

    assert "PUBLIC INSTRUCTION LINE" in r.instruction_text
    assert "CONTEXT LINE 2" in r.instruction_text
    assert "assistant reasoning" in r.body_text
    assert '> shell({"cmd": "ls -la"})' in r.body_text
    assert "file listing output" in r.body_text
    # chronological within the step: message, call, then embedded output
    assert r.body_text.index("assistant reasoning") < r.body_text.index("> shell")
    assert r.body_text.index("> shell") < r.body_text.index("file listing output")
    assert r.truncated is False


def test_render_is_deterministic_and_symmetric(tmp_path: Path) -> None:
    atif = _atif(
        [_user("INSTR"), _agent(text="a"), _agent(calls=[("sh", {"c": "1"})], outputs=["o"])]
    )
    p = _write(tmp_path, atif)
    r1 = render_trajectory("A", "t", p, 10**6)
    r2 = render_trajectory("B", "t", p, 10**6)
    assert r1.body_text == r2.body_text
    assert r1.instruction_text == r2.instruction_text
    assert r1.rendered_tokens == r2.rendered_tokens


def test_head_tail_truncation_preserves_instruction(tmp_path: Path) -> None:
    steps = [_user("FULL INSTRUCTION " + "x" * 300)] + [
        _agent(outputs=[f"line {i} " + "y" * 50]) for i in range(30)
    ]
    atif = _atif(steps)
    budget = 200  # holds a few items but not all 30
    r = render_trajectory("C", "t", _write(tmp_path, atif), budget)
    assert r.truncated is True
    # instruction survives verbatim
    assert "FULL INSTRUCTION " + "x" * 300 in r.instruction_text
    # marker present, head+tail retained
    assert "OUTPUT TRUNCATED" in r.body_text
    assert "line 0 " in r.body_text  # head from the front
    assert "line 29 " in r.body_text  # tail from the back


def test_original_vs_rendered_token_counts_recorded(tmp_path: Path) -> None:
    atif = _atif([_user("INSTR"), _agent(outputs=["x" * 400])])
    r = render_trajectory("C", "t", _write(tmp_path, atif), 10**6)
    assert r.original_tokens > 0
    assert r.rendered_tokens == estimate_tokens(r.instruction_text + "\n\n" + r.body_text)
    assert r.rendered_tokens >= r.original_tokens  # nothing dropped


def test_leakage_verifier_payload_excluded(tmp_path: Path) -> None:
    """The renderer reads only the agent trajectory; hidden tests / reward files
    in the trial's verifier/ dir must never appear in the rendered prompt."""
    trial = tmp_path / "trial"
    (trial / "agent").mkdir(parents=True)
    (trial / "verifier").mkdir()
    _write(trial / "agent", _atif([_user("TASK"), _agent(text="work", outputs=["out"])]))
    (trial / "verifier" / "reward.txt").write_text("1.0\n", encoding="utf-8")
    (trial / "verifier" / "ctrf.json").write_text(
        '{"tests":[{"name":"hidden_behaviour_check","result":"PASS"}]}', encoding="utf-8"
    )
    (trial / "verifier" / "test-stdout.txt").write_text(
        "HIDDEN_REWARD=1.0\nSECRET_TEST_EXPECT=", encoding="utf-8"
    )

    r = render_trajectory("C", "t", trial / "agent" / "trajectory.json", 10**6)
    full = r.instruction_text + "\n\n" + r.body_text
    assert "1.0" not in full
    assert "hidden_behaviour_check" not in full
    assert "HIDDEN_REWARD" not in full
    assert "SECRET_TEST_EXPECT" not in full
