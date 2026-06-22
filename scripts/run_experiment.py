#!/usr/bin/env python
"""Run the surrogate-assisted CMA-ES COCO/BBOB experiment.

Examples
--------
    # Tiny smoke matrix (defined in configs/smoke.yaml)
    python scripts/run_experiment.py --config configs/smoke.yaml

    # Full Stage 1 noiseless run
    python scripts/run_experiment.py --config configs/stage1_noiseless.yaml

    # Shard across processes/machines by variant subset
    python scripts/run_experiment.py --config configs/stage1_noiseless.yaml --models gp bnn_mc_dropout
    python scripts/run_experiment.py --config configs/stage1_noiseless.yaml --ecs lq

Re-runs skip variants already marked complete; use --force to redo them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from sacma.runner import ExperimentConfig, run_experiment  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=str, default=None, help="YAML config file.")
    ap.add_argument("--models", nargs="+", default=None, help="Subset of surrogate models.")
    ap.add_argument("--ecs", nargs="+", default=None, help="Subset of evolution controls.")
    ap.add_argument("--functions", type=str, default=None, help="COCO function spec, e.g. '1-24' or '1,8,15'.")
    ap.add_argument("--dims", type=int, nargs="+", default=None, help="Dimensions.")
    ap.add_argument("--instances", type=str, default=None, help="Instance spec, e.g. '1-15'.")
    ap.add_argument("--budget-mult", type=int, default=None, help="Budget = mult * D (default 250).")
    ap.add_argument("--seed", type=int, default=None, help="Master seed.")
    ap.add_argument("--force", action="store_true", help="Re-run variants even if marked complete.")
    args = ap.parse_args()

    raw: dict = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    # CLI overrides
    if args.models is not None:      raw["models"] = args.models
    if args.ecs is not None:         raw["ecs"] = args.ecs
    if args.functions is not None:   raw["functions"] = args.functions
    if args.dims is not None:        raw["dimensions"] = args.dims
    if args.instances is not None:   raw["instances"] = args.instances
    if args.budget_mult is not None: raw["budget_mult"] = args.budget_mult
    if args.seed is not None:        raw["seed"] = args.seed

    cfg = ExperimentConfig.from_dict(raw)
    from sacma.runner import expand_indices
    n_units = len(cfg.models) * len(cfg.ecs) * len(cfg.dimensions) * len(expand_indices(cfg.functions))
    print(f"Models: {cfg.models}")
    print(f"ECs: {cfg.ecs}")
    print(f"Functions: {cfg.functions} | Dims: {cfg.dimensions} | Instances: {cfg.instances}")
    print(f"Budget: {cfg.budget_mult} * D | Seed: {cfg.seed} | Suite: {cfg.suite_name}")
    print(f"Resume units (variant x dim x func): {n_units}\n")

    summaries = run_experiment(cfg, force=args.force)
    done = sum(1 for s in summaries if s["status"] == "complete")
    skipped = sum(1 for s in summaries if s["status"] == "skipped")
    print(f"\nFinished: {done} run, {skipped} skipped, {len(summaries)} total units.")


if __name__ == "__main__":
    main()
