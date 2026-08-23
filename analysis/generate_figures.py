"""Stage 19 analysis: reproduce metrics and emit SVG figures from the frozen DBs.

Pure-stdlib (no matplotlib) so figures regenerate identically from a fresh
clone without altering the declared environment. Reads only the experiment and
ground-truth SQLite catalogs under the frozen config's storage paths.

Metrics follow plan sections 21-22. Verifier accuracy is computed PER PAIR from
the stored per-(pair,criterion) responses (`verifications.response_path` via
`expected_scores_from_logprobs`, and `scores_path` for the discrete judge) on
polarized pairs where official rewards differ; it is never derived from the
pool-wide candidate aggregate.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from pathlib import Path

from avt.verification import SCORE_LABELS, expected_scores_from_logprobs

# --- SVG helpers ------------------------------------------------------------

_W = 720
_H = 460


def _svg_open() -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_W}" height="{_H}" '
        'font-family="DejaVu Sans, Arial, sans-serif" font-size="12">',
    ]


def _svg_close() -> list[str]:
    return ["</svg>"]


def _bar(
    s: list[str], x: float, y: float, w: float, h: float, fill: str, top: str, names: str
) -> None:
    s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}"/>')
    s.append(
        f'<text x="{x + w / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-size="13">{top}</text>'  # noqa: E501
    )
    lines = names.split("\n")
    for i, ln in enumerate(lines):
        s.append(
            f'<text x="{x + w / 2:.1f}" y="{y + h + 14 + i * 12:.1f}" text-anchor="middle" '
            f'font-size="10" fill="#444">{ln}</text>'
        )


def _line(
    s: list[str], x1: float, y1: float, x2: float, y2: float, stroke: str, dash: str = ""
) -> None:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    s.append(
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="1.4"{d}/>'  # noqa: E501
    )


def _text(
    s: list[str],
    x: float,
    y: float,
    t: str,
    size: int = 12,
    anchor: str = "start",
    weight: str = "normal",
) -> None:
    s.append(
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{t}</text>'  # noqa: E501
    )


def _err(s: list[str], x: float, cy: float, lo_px: float, hi_px: float) -> None:
    s.append(
        f'<line x1="{x:.1f}" y1="{cy - hi_px:.1f}" x2="{x:.1f}" y2="{cy + lo_px:.1f}" stroke="#222" stroke-width="1.4"/>'  # noqa: E501
    )
    s.append(
        f'<line x1="{x - 4:.1f}" y1="{cy - hi_px:.1f}" x2="{x + 4:.1f}" y2="{cy - hi_px:.1f}" stroke="#222" stroke-width="1.4"/>'  # noqa: E501
    )
    s.append(
        f'<line x1="{x - 4:.1f}" y1="{cy + lo_px:.1f}" x2="{x + 4:.1f}" y2="{cy + lo_px:.1f}" stroke="#222" stroke-width="1.4"/>'  # noqa: E501
    )


# --- bootstrap --------------------------------------------------------------


def _paired_boot(
    a: list[float], b: list[float], seed: int, n: int = 20000
) -> tuple[float, float, float]:
    rng = random.Random(seed)
    obs = [x - y for x, y in zip(a, b, strict=True)]
    k = len(obs)
    means = [sum(obs[i] for i in rng.choices(range(k), k=k)) / k for _ in range(n)]
    means.sort()
    return (sum(obs) / k, means[int(0.025 * n)], means[int(0.975 * n)])


def _mean_boot(values: list[float], seed: int, n: int = 20000) -> tuple[float, float, float]:
    """Task-level mean bootstrap over per-task fractions (no binarization)."""
    rng = random.Random(seed)
    k = len(values)
    means = [sum(values[i] for i in rng.choices(range(k), k=k)) / k for _ in range(n)]
    means.sort()
    return (sum(values) / k, means[int(0.025 * n)], means[int(0.975 * n)])


# --- data -------------------------------------------------------------------


def _per_pair_verifier(exp: sqlite3.Connection) -> tuple[dict, dict]:
    """Per-(pair,criterion) continuous and discrete scores keyed by pair then candidate."""
    cont: dict[str, dict[str, list[float]]] = {}
    disc: dict[str, dict[str, list[float]]] = {}
    for pid, disp, resp, sp in exp.execute(
        "select pair_id, display_order, response_path, scores_path "
        "from verifications where status='SUCCEEDED'"
    ).fetchall():
        a, b = disp.split("+")
        try:
            r = json.loads(Path(resp).read_text(encoding="utf-8"))
            ch = r.get("choices") or [{}]
            lp = (ch[0].get("logprobs") or {}) if isinstance(ch[0], dict) else {}
            ra, rb = expected_scores_from_logprobs(list(lp.get("content") or []), SCORE_LABELS)
        except Exception:
            ra = rb = None
        cont.setdefault(pid, {}).setdefault(a, []).append(ra)
        cont.setdefault(pid, {}).setdefault(b, []).append(rb)
        try:
            s = json.loads(Path(sp).read_text(encoding="utf-8"))
            sa, sb = s["score_a"], s["score_b"]
        except Exception:
            sa = sb = None
        disc.setdefault(pid, {}).setdefault(a, []).append(sa)
        disc.setdefault(pid, {}).setdefault(b, []).append(sb)
    return cont, disc


def load(config_path: str, root: str) -> dict:
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    exp = sqlite3.connect(Path(cfg["storage"]["metadata_db"]).expanduser())
    gt = sqlite3.connect(Path(cfg["storage"]["ground_truth_db"]).expanduser())
    reward = {
        r[0]: r[1]
        for r in gt.execute("select candidate_id,reward from official_results").fetchall()
    }
    agg = {
        r[0]: r[1]
        for r in exp.execute("select candidate_id,aggregate_raw from evaluation").fetchall()
    }

    tasks: dict[str, dict[str, float]] = {}
    for cid, tid in exp.execute("select candidate_id, task_id from candidates").fetchall():
        tasks.setdefault(tid, {})[cid] = reward.get(cid, 0.0)

    sel: dict[str, dict[str, str]] = {}
    for tid, scfg, res in exp.execute(
        "select task_id, selector_config, result from rankings"
    ).fetchall():
        s = json.loads(scfg)["selector"]
        sel.setdefault(s, {})[tid] = json.loads(res)["ranking"][0]["candidate_id"]

    cont, disc = _per_pair_verifier(exp)
    pol_total = agree_cont = agree_disc = 0
    for pid, ca_, cb_ in exp.execute(
        "select pair_id, candidate_a, candidate_b from pairs"
    ).fetchall():
        ra_, rb_ = reward.get(ca_), reward.get(cb_)
        if not isinstance(ra_, (int, float)) or not isinstance(rb_, (int, float)) or ra_ == rb_:
            continue
        pol_total += 1
        passer, failer = (ca_, cb_) if ra_ > rb_ else (cb_, ca_)

        def mean(store, passer_, failer_, pid_):
            m = {}
            for cid in (passer_, failer_):
                v = store.get(pid_, {}).get(cid, [])
                if v and all(x is not None for x in v):
                    m[cid] = sum(v) / len(v)
            return m

        mc = mean(cont, passer, failer, pid)
        if passer in mc and failer in mc:
            agree_cont += int(mc[passer] > mc[failer])
        md = mean(disc, passer, failer, pid)
        if passer in md and failer in md:
            agree_disc += int(md[passer] > md[failer])

    tot_in = tot_out = 0
    lats: list[float] = []
    from datetime import datetime

    for f in Path(root).rglob("result.json"):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        s = r.get("stats") or {}
        if "n_input_tokens" not in s:
            continue
        tot_in += s.get("n_input_tokens") or 0
        tot_out += s.get("n_output_tokens") or 0
        if r.get("started_at") and r.get("finished_at"):
            try:
                t0 = datetime.fromisoformat(r["started_at"])
                t1 = datetime.fromisoformat(r["finished_at"])
                lats.append((t1 - t0).total_seconds())
            except Exception:
                pass

    return {
        "exp": exp,
        "reward": reward,
        "agg": agg,
        "tasks": tasks,
        "sel": sel,
        "tlist": sorted((sel.get("continuous") or {}).keys()),
        "verifier": (pol_total, agree_cont, agree_disc),
        "compute": (tot_in, tot_out, lats),
    }


# --- run --------------------------------------------------------------------


def run(config_path: str, root: str, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    d = load(config_path, root)
    tlist = d["tlist"]
    st = d["sel"]
    reward, tasks = d["reward"], d["tasks"]

    def topval(s, t):
        return tasks[t].get(st[s][t], 0.0)

    er = [sum(tasks[t].values()) / 5.0 for t in tlist]
    pr = {
        s: _mean_boot([topval(s, t) for t in tlist], seed=11)
        for s in ["random", "discrete", "continuous"]
    }
    # Pool base pass and uniform-expected-random uncertainty bootstrap the 25
    # per-TASK pass fractions (plan 22: bootstrap tasks), never individual
    # candidates, and do not binarize fractional values.
    pool = _mean_boot(er, seed=11)
    cont = [topval("continuous", t) for t in tlist]
    disc = [topval("discrete", t) for t in tlist]
    diff_cr = _paired_boot(cont, er, seed=42)
    diff_dr = _paired_boot(disc, er, seed=42)
    diff_cd = _paired_boot(cont, disc, seed=42)
    pol_total, agree_cont, agree_disc = d["verifier"]
    tot_in, tot_out, lats = d["compute"]

    stats = {
        "pool_base_pass": pool,
        "selector_pass": pr,
        "diff_continuous_expected_random": diff_cr,
        "diff_discrete_expected_random": diff_dr,
        "diff_continuous_discrete": diff_cd,
        "expected_random_mean": sum(er) / len(er),
        "verifier": {
            "polarized_pairs": pol_total,
            "per_pair_continuous_accuracy": agree_cont / pol_total if pol_total else None,
            "per_pair_discrete_accuracy": agree_disc / pol_total if pol_total else None,
        },
        "compute": {
            "total_input_tokens": tot_in,
            "total_output_tokens": tot_out,
            "candidates_with_stats": len(lats),
            "gen_latency_median": sorted(lats)[len(lats) // 2] if lats else None,
        },
    }

    # ---- figure 1: top-pass rate by selector + pool base ----
    svg = _svg_open()
    _text(
        svg,
        20,
        32,
        "Selected top-pass rate by local selector (25 tasks, 95% CI)",
        14,
        weight="bold",
    )
    base_y = _H - 70
    groups = [
        ("random\nrealized", "random", "#f58518"),
        ("random\nexpected", None, "#bbbbbb"),
        ("discrete", "discrete", "#54a24b"),
        ("continuous", "continuous", "#4c78a8"),
    ]
    n = len(groups)
    bw = _W * 0.12
    gap = (_W - 60) / n
    pool_rate = pool[0]
    ph = 250.0
    for i, (label, key, fill) in enumerate(groups):
        if key is None:
            m, lo, hi = stats["expected_random_mean"], pool[1], pool[2]
        else:
            m, lo, hi = pr[key]
        x = 40 + i * gap + (gap - bw) / 2
        h = m / 1.0 * ph
        y = base_y - h
        y_e = base_y - m * ph
        _bar(svg, x, y, bw, h, fill, f"{m:.0%}", label)
        _err(svg, x + bw / 2, y_e, (m - lo) * ph, (hi - m) * ph)
    _line(svg, 30, base_y, _W - 20, base_y, "#222")
    _line(svg, 30, base_y - pool_rate * ph, _W - 20, base_y - pool_rate * ph, "#888", dash="4,3")
    _text(svg, _W - 20, base_y - pool_rate * ph - 4, f"pool base {pool_rate:.0%}", 10, anchor="end")
    svg.extend(_svg_close())
    (outdir / "fig-selector-pass-rate.svg").write_text("\n".join(svg), encoding="utf-8")

    # ---- figure 2: bootstrap difference histogram (continuous - expected random) ----
    rng = random.Random(42)
    obs = [c - e for c, e in zip(cont, er, strict=True)]
    diffs = [
        sum(obs[i] for i in rng.choices(range(len(obs)), k=len(obs))) / len(obs)
        for _ in range(20000)
    ]
    lo, hi = diff_cr[1], diff_cr[2]
    svg = _svg_open()
    _text(svg, 20, 32, "Bootstrap: continuous - expected-random selected reward", 14, weight="bold")
    bins = 32
    lo_v, hi_v = min(diffs), max(diffs)
    hist = [0] * bins
    for v in diffs:
        idx = min(bins - 1, int((v - lo_v) / (hi_v - lo_v) * bins))
        hist[idx] += 1
    mx = max(hist)
    bw = (_W - 60) / bins
    for i, c in enumerate(hist):
        h = c / mx * 250
        x = 30 + i * bw
        v = lo_v + (i + 0.5) / bins * (hi_v - lo_v)
        fill = "#4c78a8" if lo <= v <= hi else "#dddddd"
        svg.append(
            f'<rect x="{x:.1f}" y="{base_y - h:.1f}" width="{bw + 0.5:.1f}" height="{h:.1f}" fill="{fill}"/>'  # noqa: E501
        )
    _text(
        svg,
        30,
        base_y - 260,
        f"mean {diff_cr[0]:.3f}  95% CI [{diff_cr[1]:.3f},{diff_cr[2]:.3f}]",
        12,
    )
    for v in (lo, hi):
        lx = 30 + (v - lo_v) / (hi_v - lo_v) * (_W - 60)
        _line(svg, lx, base_y, lx, base_y - 244, "#c22", dash="3,3")
    _line(svg, 30, base_y, _W - 30, base_y, "#222")
    svg.extend(_svg_close())
    (outdir / "fig-continuous-vs-random-bootstrap.svg").write_text("\n".join(svg), encoding="utf-8")

    # ---- figure 3: verifier aggregate score separation by official pass/fail ----
    passes = [d["agg"][c] for c, r in reward.items() if r > 0]
    fails = [d["agg"][c] for c, r in reward.items() if r == 0]
    allv = passes + fails
    vmin, vmax = min(allv), max(allv)

    def scale(v):
        return 30 + (v - vmin) / (vmax - vmin) * (_W - 60)

    svg = _svg_open()
    _text(svg, 20, 32, "Verifier aggregate expected score by official reward", 14, weight="bold")

    def strip(vals, yc, label):
        drawn: list[tuple[float, float]] = []
        xs = sorted(scale(v) for v in vals)
        for x in xs:
            y = yc
            while any(abs(x - dx) < 9 and abs(y - dy) < 9 for dx, dy in drawn):
                y += 8
            drawn.append((x, y))
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="#4c78a8" opacity="0.7"/>')
        sm = sum(vals) / len(vals)
        _line(svg, scale(sm), yc - 56, scale(sm), yc + 56, "#c22", dash="3,3")
        _text(svg, scale(sm), yc - 62, f"mean {sm:.2f}", 11, anchor="middle")
        _text(svg, 20, yc - 74, label, 12, weight="bold")

    strip(passes, 150, f"official PASS (n={len(passes)})")
    strip(fails, 330, f"official FAIL (n={len(fails)})")
    _line(svg, 30, _H - 30, _W - 30, _H - 30, "#222")
    _text(svg, 30, _H - 16, f"{vmin:.2f}", 10)
    _text(svg, _W - 30, _H - 16, f"{vmax:.2f}", 10, anchor="end")
    svg.extend(_svg_close())
    (outdir / "fig-verifier-score-separation.svg").write_text("\n".join(svg), encoding="utf-8")

    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/frozen_main.yaml")
    ap.add_argument("--root", default="/home/workbench/avt-data/main-v1")
    ap.add_argument("--outdir", default="results/figures")
    a = ap.parse_args()
    stats = run(a.config, a.root, Path(a.outdir))
    print(json.dumps(stats, indent=2, default=str))
    (Path(a.outdir).parent / "stats.json").write_text(
        json.dumps(stats, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
