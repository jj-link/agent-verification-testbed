"""Command-line interface for the Agent Verification Testbed."""

from __future__ import annotations

import argparse
from pathlib import Path

__all__ = ["main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="avt", description="Agent Verification Testbed")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    sub.add_parser("doctor", help="run endpoint diagnostics")

    p_sel = sub.add_parser("select-tasks", help="select smoke/pilot/main tasks")
    p_sel.add_argument("--config", required=True, help="experiment config path")
    p_sel.add_argument("--smoke", type=int, default=2)
    p_sel.add_argument("--pilot", type=int, default=8)
    p_sel.add_argument("--main", type=int, default=25)

    p_gen = sub.add_parser("generate", help="generate candidates for the experiment")
    p_gen.add_argument("--config", required=True, help="experiment config path")

    p_pairs = sub.add_parser("build-pairs", help="build frozen candidate pairs")
    p_pairs.add_argument("--config", required=True, help="experiment config path")

    p_verify = sub.add_parser("verify-pairs", help="run the discrete judge over pairs")
    p_verify.add_argument("--config", required=True, help="experiment config path")

    p_exp = sub.add_parser("expected-scores", help="compute continuous expected scores")
    p_exp.add_argument("--config", required=True, help="experiment config path")

    p_eval = sub.add_parser("evaluate", help="compute three-criterion aggregate scores")
    p_eval.add_argument("--config", required=True, help="experiment config path")

    p_rank = sub.add_parser("rank", help="rank each task pool (round-robin BT)")
    p_rank.add_argument("--config", required=True, help="experiment config path")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "doctor":
        from avt.doctor import format_results, run_doctor

        doctor_results = run_doctor()
        print(format_results(doctor_results))
        return 0 if all(r.ok for r in doctor_results) else 1

    if args.command == "select-tasks":
        from avt.config import load_config
        from avt.selection import fetch_task_pool, select_tasks, write_task_file

        cfg = load_config(args.config)
        commit = str(cfg.upstream.get("terminal_bench_commit", ""))
        if not commit:
            parser.error("config missing upstream.terminal_bench_commit")
        pool = fetch_task_pool(commit=commit)
        if len(pool) < args.main:
            parser.error(f"pool has {len(pool)} tasks, need {args.main}")
        selected = select_tasks(
            pool, smoke=args.smoke, pilot=args.pilot, main=args.main, seed=cfg.experiment.seed
        )
        base = Path(cfg.experiment.task_file).parent
        for name, tasks in selected.items():
            path = write_task_file(base / f"{name}_tasks.txt", tasks)
            print(f"{name}: {len(tasks)} tasks -> {path}")
        return 0

    if args.command == "generate":
        from avt.config import load_config
        from avt.generation import GenerationService

        cfg = load_config(args.config)
        service = GenerationService(cfg, Path.cwd())
        gen_results = service.generate_all()
        failed = [r for r in gen_results if r.reward is None]
        for r in gen_results:
            state = "ok" if r.reward is not None else "FAILED"
            print(f"[{state}] {r.task_id} attempt={r.attempt_index} reward={r.reward}")
        print(f"total={len(gen_results)} ok={len(gen_results) - len(failed)} failed={len(failed)}")
        return 0 if not failed else 1

    if args.command == "build-pairs":
        from avt.config import load_config
        from avt.pairs import PairBuilder

        cfg = load_config(args.config)
        builder = PairBuilder(cfg, Path.cwd())
        pair_ids = builder.build()
        print(f"built {len(pair_ids)} pairs")
        return 0

    if args.command == "verify-pairs":
        from avt.config import load_config
        from avt.verification import DiscreteJudge

        cfg = load_config(args.config)
        judge = DiscreteJudge(cfg, Path.cwd())
        scores = judge.verify_all()
        malformed = [s for s in scores if s.status != "SUCCEEDED"]
        with judge.catalog.connect() as scoped:
            stage = scoped.get_experiment_stage(judge.exp)
        print(
            f"verified {len(scores)} pair-scores this run "
            f"({len(scores) - len(malformed)} succeeded, {len(malformed)} malformed/failed); "
            f"experiment stage={stage}"
        )
        # VERIFIED requires full usable coverage; a FAILED verification keeps the
        # stage at VERIFYING and exits nonzero (plan 14: fail visibly).
        return 0 if stage == "VERIFIED" else 1

    if args.command == "expected-scores":
        from avt.config import load_config
        from avt.expected import ContinuousVerifier

        cfg = load_config(args.config)
        recs = ContinuousVerifier(cfg, Path.cwd()).compute()
        print(f"expected-score records: {len(recs)}")
        return 0

    if args.command == "evaluate":
        from avt.config import load_config
        from avt.evaluation import Evaluator

        cfg = load_config(args.config)
        records = Evaluator(cfg, Path.cwd()).evaluate()
        print(f"evaluated {len(records)} candidates")
        return 0

    if args.command == "rank":
        from avt.config import load_config
        from avt.ranking import RoundRobinRanker

        cfg = load_config(args.config)
        rankings = RoundRobinRanker(cfg, Path.cwd()).rank_all()
        for rec in rankings:
            top = rec.ranking[0]
            print(f"{rec.task_id}: top={top.candidate_id} utility={top.utility:.4f}")
        print(f"ranked {len(rankings)} tasks")
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2
