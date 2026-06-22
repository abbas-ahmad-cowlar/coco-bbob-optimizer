#!/usr/bin/env python
"""Post-process COCO results into the protocol's tables, statistics, and plots.

Reads:
    exdata/<variant>/            COCO output (via cocopp) -> Delta-mu-f, convergence
    results/diagnostics/*.jsonl  EC diagnostics -> ablation (surrogate savings etc.)

Writes (results/processed/):
    long_table.csv               one row per (variant, func, dim, instance, FE/D)
    overall_table.csv            mean Delta-mu-f per variant @50 and @250 FE/D
    groups_{50,250}FEbyD.csv     mean Delta-mu-f per variant x 12-function-group
    winmatrix_{50,250}FEbyD.csv  pairwise %-win matrices
    wilcoxon_{50,250}FEbyD.csv   Wilcoxon signed-rank + Holm-corrected p-values
    diagnostics_summary.csv      per-variant ablation summary
    plots/convergence_fXX_dYY.png  median convergence with quartile bands

Usage:
    python scripts/analyze.py [--exdata exdata] [--diag results/diagnostics]
                              [--out results/processed] [--budget-mult 250]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from sacma.baselines import BASELINES, fetch_baselines  # noqa: E402
from sacma.metrics import (build_baseline_table, build_long_table,  # noqa: E402
                           group_table, overall_table)
from sacma.plots import plot_convergence  # noqa: E402
from sacma.stats import pairwise_win_matrix, wilcoxon_holm  # noqa: E402


def discover_variants(diag_dir: Path) -> list[str]:
    return sorted(p.stem for p in diag_dir.glob("*.done"))


def aggregate_diagnostics(diag_dir: Path, variants: list[str]) -> pd.DataFrame:
    rows = []
    for v in variants:
        jl = diag_dir / f"{v}.jsonl"
        if not jl.exists():
            continue
        recs = [json.loads(l) for l in jl.read_text(encoding="utf-8").splitlines() if l.strip()]
        df = pd.DataFrame(recs)
        rows.append({
            "variant": v,
            "n_runs": len(df),
            "mean_real_eval_fraction": df["real_eval_fraction"].mean(),
            "mean_evals_saved": df["n_real_evals_saved"].mean(),
            "mean_surrogate_gens": df["n_surrogate_generations"].mean(),
            "mean_direct_gens": df["n_direct_generations"].mean(),
            "mean_rmse": df["mean_rmse"].dropna().mean(),
            "target_hit_rate": df["target_hit"].mean(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exdata", default="exdata")
    ap.add_argument("--diag", default="results/diagnostics")
    ap.add_argument("--out", default="results/processed")
    ap.add_argument("--budget-mult", type=int, default=250)
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[50, 250])
    ap.add_argument("--baselines", nargs="*", default=None,
                    help=f"Include archived baselines in tables/plots. No args = all "
                         f"({', '.join(BASELINES)}); or list a subset.")
    ap.add_argument("--cocopp", action="store_true",
                    help="Also run cocopp.main for the standard ECDF HTML report.")
    args = ap.parse_args()

    exdata = _REPO / args.exdata
    diag_dir = _REPO / args.diag
    out = _REPO / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)

    variants = discover_variants(diag_dir)
    if not variants:
        print(f"No completed variants found in {diag_dir}")
        return
    print(f"Variants ({len(variants)}): {variants}")

    checkpoints = tuple(args.checkpoints)

    # ---- Fetch baselines (optional) ------------------------------------
    baseline_paths: dict = {}
    if args.baselines is not None:
        names = args.baselines or list(BASELINES)
        print(f"Fetching baselines: {names}")
        baseline_paths = fetch_baselines(names)

    # ---- Master table + Delta-mu-f aggregations -------------------------
    long_var = build_long_table(exdata, variants, fe_per_d=checkpoints)   # variants only
    long_all = long_var
    if baseline_paths:
        long_base = build_baseline_table(baseline_paths, fe_per_d=checkpoints)
        # Fair comparison: keep only the (function, dimension) cells our variants ran.
        present = set(map(tuple, long_var[["function", "dimension"]].drop_duplicates().to_numpy()))
        keep = [(f, d) in present for f, d in zip(long_base.function, long_base.dimension)]
        long_base = long_base[keep]
        long_all = pd.concat([long_var, long_base], ignore_index=True)
    long_all.to_csv(out / "long_table.csv", index=False)

    # Overall + group tables include baselines for context.
    overall = overall_table(long_all)
    overall.to_csv(out / "overall_table.csv")
    print("\n=== Overall mean Delta-mu-f (variants + baselines) ===")
    print(overall.round(4).to_string())

    for feD in checkpoints:
        group_table(long_all, feD).to_csv(out / f"groups_{feD}FEbyD.csv")
        # Paired stats stay among our instance-matched variants only.
        pairwise_win_matrix(long_var, feD).to_csv(out / f"winmatrix_{feD}FEbyD.csv")
        wilcoxon_holm(long_var, feD).to_csv(out / f"wilcoxon_{feD}FEbyD.csv", index=False)

    # ---- Ablation diagnostics ------------------------------------------
    diag = aggregate_diagnostics(diag_dir, variants)
    diag.to_csv(out / "diagnostics_summary.csv", index=False)
    print("\n=== Diagnostics (surrogate usage) ===")
    print(diag.round(3).to_string(index=False))

    # ---- Convergence plots (variants + baselines) ----------------------
    sources = {v: str(exdata / v) for v in variants}
    sources.update(baseline_paths)
    pairs = long_var[["function", "dimension"]].drop_duplicates().itertuples(index=False)
    n_plots = 0
    for func, dim in pairs:
        p = plot_convergence(sources, int(func), int(dim),
                             out / "plots" / f"convergence_f{int(func):02d}_d{int(dim):02d}.png",
                             budget_feD=args.budget_mult)
        if p:
            n_plots += 1
    print(f"\nWrote tables to {out} and {n_plots} convergence plots to {out/'plots'}")

    # ---- Standard cocopp ECDF report (optional) ------------------------
    if args.cocopp:
        import cocopp
        argstr = [str(exdata / v) for v in variants] + list(baseline_paths.values())
        print(f"\nRunning cocopp.main on {len(argstr)} datasets (ECDF HTML -> ppdata/) ...")
        cocopp.main(" ".join(argstr))


if __name__ == "__main__":
    main()
