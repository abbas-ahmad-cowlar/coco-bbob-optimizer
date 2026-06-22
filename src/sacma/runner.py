"""COCO/BBOB experiment runner — resume-safe, unit = (variant, dimension, function).

For each *unit* (one surrogate model x evolution control x dimension x function,
across all instances) this:
  1. attaches a cocoex ``Observer`` and runs IPOP-CMA-ES on every instance,
     producing standard COCO output in ``exdata/<variant>__D<dim>__f<func>``;
  2. records per-run EC diagnostics (surrogate savings, RMSE, surr/direct gens)
     to ``results/diagnostics/<unit>.jsonl`` — info COCO does not log but the
     ablation analysis needs.

Resume granularity is the unit (~15 short runs). A finished unit is marked with
``<unit>.done`` and skipped on re-run; a VM dying mid-run therefore loses at most
one unit (minutes), and rerunning continues from the last completed unit. This is
what makes the Colab workflow survivable (see notebooks/colab_runbook.ipynb).

Units are ordered dimension-ascending so the fast low-D results land first.
Each (function, dimension) maps to exactly one unit folder, so the analysis layer
loads them directly with no instance-splitting.

Budget follows the protocol: 250 * D true evaluations per problem. Early-stop at
COCO's 1e-8 target is OFF by default so Delta-mu-f is measured to 1e-13.
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


def expand_indices(spec) -> list[int]:
    """Expand a function/dimension spec into a list of ints.

    Accepts a list/tuple, or a string like '1-24' or '1,8,15' or '1-5,10,20'.
    """
    if isinstance(spec, (list, tuple)):
        return [int(x) for x in spec]
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


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


def unit_name(vid: str, dim: int, func: int) -> str:
    """COCO-safe folder/marker name for one (variant, dim, function) unit."""
    return f"{vid}__D{dim}__f{func}"


def parse_unit(name: str) -> tuple[str, int, int]:
    """Inverse of unit_name -> (variant_id, dim, func)."""
    vid, rest = name.split("__D", 1)
    dim_s, func_s = rest.split("__f", 1)
    return vid, int(dim_s), int(func_s)


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
    early_stop: bool = True     # once the 1e-8 target is hit, stop launching new IPOP
                                # restarts (after the solving restart converges to ~1e-14).
                                # Delta-mu-f-preserving (best-so-far cannot improve once
                                # solved) and skips the wasteful post-solve restart loop.
                                # Set False to force the full 250*D budget on every problem.
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


def run_unit(model: str, ec: str, dim: int, func: int, cfg: ExperimentConfig,
             *, force: bool = False, verbose: bool = True) -> dict:
    """Run one (variant, dim, function) unit over all instances. Resume-safe."""
    vid = variant_id(model, ec)
    uid = unit_name(vid, dim, func)
    diag_dir = _REPO / cfg.diag_dir
    diag_dir.mkdir(parents=True, exist_ok=True)
    done_marker = diag_dir / f"{uid}.done"
    jsonl_path = diag_dir / f"{uid}.jsonl"
    ex_root = _REPO / cfg.exdata_dir

    if done_marker.exists() and not force:
        return {"unit": uid, "status": "skipped"}

    # Fresh start: clear any partial COCO folder / diagnostics for this unit.
    for stale in ex_root.glob(f"{uid}*"):
        shutil.rmtree(stale, ignore_errors=True)
    jsonl_path.unlink(missing_ok=True)

    prev_cwd = Path.cwd()
    os.chdir(_REPO)
    try:
        suite = cocoex.Suite(cfg.suite_name, "", suite_filter([func], [dim], cfg.instances))
        observer = cocoex.Observer(cfg.suite_name, f"result_folder: {uid} algorithm_name: {vid}")
        n_runs = 0
        t0 = time.time()
        jf = jsonl_path.open("w", encoding="utf-8")
        for problem in suite:
            inst = problem.id_instance
            budget = cfg.budget_mult * dim
            seed = run_seed(cfg.seed, model, ec, func, dim, inst)
            problem.observe_with(observer)
            surrogate = make_surrogate(model, random_state=seed)
            opt = IPOPSurrogateCMAES(surrogate=surrogate, ec_type=ec, random_state=seed)
            done = (lambda p=problem: p.final_target_hit) if cfg.early_stop else None
            run = opt.run(problem, problem.lower_bounds, problem.upper_bounds,
                          budget=budget, is_done=done)
            rec = _diag_summary(run, model, ec, func, dim, inst, seed, budget,
                                bool(problem.final_target_hit))
            jf.write(json.dumps(rec) + "\n")
            jf.flush()
            n_runs += 1
        jf.close()
        del observer  # flush COCO files
        done_marker.write_text(
            json.dumps({"unit": uid, "n_runs": n_runs, "seconds": round(time.time() - t0, 1)}),
            encoding="utf-8",
        )
    finally:
        os.chdir(prev_cwd)
    return {"unit": uid, "status": "complete", "n_runs": n_runs}


def iter_units(cfg: ExperimentConfig):
    """Yield (model, ec, dim, func) units, ordered dimension-ascending."""
    funcs = expand_indices(cfg.functions)
    for dim in cfg.dimensions:
        for model in cfg.models:
            if model not in SURROGATES:
                raise KeyError(f"unknown model '{model}'")
            for ec in cfg.ecs:
                if ec not in EC_TYPES:
                    raise KeyError(f"unknown ec '{ec}'")
                for func in funcs:
                    yield model, ec, dim, func


def run_experiment(cfg: ExperimentConfig, *, force: bool = False,
                   verbose: bool = True) -> list[dict]:
    """Run all selected units (variant x dim x function), resume-aware."""
    units = list(iter_units(cfg))
    total = len(units)
    summaries = []
    t0 = time.time()
    done = skipped = 0
    for i, (model, ec, dim, func) in enumerate(units, 1):
        s = run_unit(model, ec, dim, func, cfg, force=force, verbose=verbose)
        summaries.append(s)
        if s["status"] == "complete":
            done += 1
        else:
            skipped += 1
        if verbose and (i % 20 == 0 or i == total):
            print(f"[{i}/{total}] {done} run, {skipped} skipped "
                  f"({time.time()-t0:.0f}s) last={s['unit']}", flush=True)
    return summaries
