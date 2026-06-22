"""COCO/BBOB experiment runner.

For each algorithm variant (surrogate model x evolution control) this:
  1. attaches a cocoex ``Observer`` and runs IPOP-CMA-ES over the filtered suite,
     producing standard COCO output in ``exdata/<variant>`` (read later by cocopp);
  2. records per-run evolution-control diagnostics (surrogate savings, RMSE,
     surrogate/direct generations) to ``results/diagnostics/<variant>.jsonl`` —
     information COCO does not log but the ablation analysis needs.

Resume granularity is the *variant* (the COCO-idiomatic unit: one algorithm == one
result folder spanning all functions/dimensions/instances). A finished variant is
marked with ``<variant>.done`` and skipped on re-run. To parallelise, launch
several processes over disjoint ``--models`` / ``--ecs`` subsets; each variant
writes to its own folder, so there are no collisions.

Budget accounting follows the protocol: 250 * D true evaluations per problem,
early-stopping when COCO's final target (1e-8) is hit.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import cocoex

from .optimizer import IPOPSurrogateCMAES
from .surrogates import EC_TYPES, SURROGATES, make_surrogate, variant_id

_REPO = Path(__file__).resolve().parents[2]


def _norm_index_spec(spec) -> str:
    """Normalise a function/instance spec to COCO's string form ('1-24' or '1,3,5')."""
    if isinstance(spec, str):
        return spec
    return ",".join(str(int(v)) for v in spec)


def suite_filter(functions, dimensions, instances) -> str:
    """Build a cocoex Suite options string."""
    dims = ",".join(str(int(d)) for d in dimensions)
    return (
        f"function_indices: {_norm_index_spec(functions)} "
        f"dimensions: {dims} "
        f"instance_indices: {_norm_index_spec(instances)}"
    )


def run_seed(base_seed: int, model: str, ec: str, func: int, dim: int, inst: int) -> int:
    """Deterministic, collision-resistant per-run seed (reproducible & reported)."""
    mi = list(SURROGATES).index(model)
    ei = list(EC_TYPES).index(ec)
    h = (((((base_seed * 7 + mi) * 5 + ei) * 31 + func) * 41 + dim) * 97 + inst)
    return int(h % (2**31 - 1))


@dataclass
class ExperimentConfig:
    suite_name: str = "bbob"               # noiseless BBOB
    models: list[str] = field(default_factory=lambda: list(SURROGATES))
    ecs: list[str] = field(default_factory=lambda: list(EC_TYPES))
    functions: object = "1-24"
    dimensions: list[int] = field(default_factory=lambda: [2, 3, 5, 10, 20])
    instances: object = "1-15"
    budget_mult: int = 250
    seed: int = 42
    exdata_dir: str = "exdata"
    diag_dir: str = "results/diagnostics"

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


def _diag_summary(run: dict, model: str, ec: str, func: int, dim: int,
                  inst: int, seed: int, budget: int, target_hit: bool) -> dict:
    d = run["diagnostics"]
    n_evals = run["n_evals"]
    saved = d["n_real_evals_saved"]
    rmse = d["surrogate_rmse_history"]
    p_real = d["p_real_history"]
    return {
        "variant": variant_id(model, ec),
        "model": model, "ec": ec,
        "function": func, "dimension": dim, "instance": inst,
        "seed": seed, "budget": budget,
        "best_y": run["best_y"], "n_evals": n_evals, "target_hit": bool(target_hit),
        "n_surrogate_generations": d["n_surrogate_generations"],
        "n_direct_generations": d["n_direct_generations"],
        "n_real_evals_saved": saved,
        "real_eval_fraction": n_evals / max(1, n_evals + saved),
        "mean_rmse": float(np.mean(rmse)) if rmse else None,
        "mean_p_real": float(np.mean(p_real)) if p_real else None,
        "n_restarts": d["n_restarts"],
        "popsize_schedule": d["popsize_schedule"],
    }


def run_variant(model: str, ec: str, cfg: ExperimentConfig,
                *, force: bool = False, verbose: bool = True) -> dict:
    """Run one variant over the whole filtered suite. Returns a small summary."""
    vid = variant_id(model, ec)
    diag_dir = _REPO / cfg.diag_dir
    diag_dir.mkdir(parents=True, exist_ok=True)
    done_marker = diag_dir / f"{vid}.done"
    jsonl_path = diag_dir / f"{vid}.jsonl"
    # COCO always writes under an "exdata/" base in the current directory and does
    # not create nested result_folder paths, so we run from the repo root and pass
    # the bare variant name as result_folder -> exdata/<vid>.
    ex_root = _REPO / cfg.exdata_dir
    ex_folder = ex_root / vid

    if done_marker.exists() and not force:
        if verbose:
            print(f"[skip] {vid} already complete")
        return {"variant": vid, "status": "skipped"}

    # Fresh start: clear any partial COCO folder / diagnostics so COCO does not
    # append to a stale folder and diagnostics do not duplicate.
    for stale in ex_root.glob(f"{vid}*"):
        shutil.rmtree(stale, ignore_errors=True)
    jsonl_path.unlink(missing_ok=True)

    prev_cwd = Path.cwd()
    os.chdir(_REPO)
    try:
        suite = cocoex.Suite(cfg.suite_name, "",
                             suite_filter(cfg.functions, cfg.dimensions, cfg.instances))
        observer = cocoex.Observer(cfg.suite_name,
                                  f"result_folder: {vid} algorithm_name: {vid}")
        n_runs = 0
        t0 = time.time()
        jf = jsonl_path.open("w", encoding="utf-8")
        for problem in suite:
            func, dim, inst = problem.id_function, problem.dimension, problem.id_instance
            budget = cfg.budget_mult * dim
            seed = run_seed(cfg.seed, model, ec, func, dim, inst)

            problem.observe_with(observer)
            surrogate = make_surrogate(model, random_state=seed)
            opt = IPOPSurrogateCMAES(surrogate=surrogate, ec_type=ec, random_state=seed)
            run = opt.run(
                problem, problem.lower_bounds, problem.upper_bounds,
                budget=budget, is_done=lambda p=problem: p.final_target_hit,
            )
            rec = _diag_summary(run, model, ec, func, dim, inst, seed, budget,
                                bool(problem.final_target_hit))
            jf.write(json.dumps(rec) + "\n")
            jf.flush()
            n_runs += 1
            if verbose and n_runs % 25 == 0:
                print(f"[{vid}] {n_runs} runs ({time.time()-t0:.0f}s)")

        jf.close()
        del observer  # flush COCO files
        done_marker.write_text(
            json.dumps({"variant": vid, "n_runs": n_runs,
                        "seconds": round(time.time() - t0, 1)}),
            encoding="utf-8",
        )
        if verbose:
            print(f"[done] {vid}: {n_runs} runs in {time.time()-t0:.0f}s -> {ex_folder}")
    finally:
        os.chdir(prev_cwd)
    return {"variant": vid, "status": "complete", "n_runs": n_runs}


def run_experiment(cfg: ExperimentConfig, *, force: bool = False,
                   verbose: bool = True) -> list[dict]:
    """Run all selected variants (model x ec), resume-aware."""
    summaries = []
    for model in cfg.models:
        if model not in SURROGATES:
            raise KeyError(f"unknown model '{model}'")
        for ec in cfg.ecs:
            if ec not in EC_TYPES:
                raise KeyError(f"unknown ec '{ec}'")
            summaries.append(run_variant(model, ec, cfg, force=force, verbose=verbose))
    return summaries
