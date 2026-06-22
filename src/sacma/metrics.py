"""Δµf metric and aggregation over COCO/BBOB results.

Δµf (from the experiment instructions, the experiment protocol §16):
    For a fixed budget, Δµf is the Lebesgue measure of the log-transform of the
    achieved target-precision subset within [1e-13, 1e7], normalised so Δµf = 1
    when every target in that interval is reached.

Because best-so-far precision decreases monotonically, the achieved subset is the
contiguous interval [delta_B, 1e7] where delta_B is the best precision reached
within the budget. Hence:

    Δµf = clip( (log10(1e7) - log10(max(delta_B, 1e-13))) / 20 , 0, 1 )

Precision trajectories come from cocopp's ``DataSet.funvals`` (col 0 = evals,
remaining cols = best precision per instance), which retains full sub-1e-8
resolution — unlike ``detEvals``, whose target grid cocopp caps at 1e-8.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .surrogates import parse_variant

TARGET_HI = 1e7
TARGET_LO = 1e-13
LOG_HI = 7.0                       # log10(TARGET_HI)
LOG_RANGE = 20.0                   # log10(1e7) - log10(1e-13)

# The 12 BBOB function groups — noiseless subset (Stage 1).
NOISELESS_GROUPS: dict[str, list[int]] = {
    "f1-5 separable": [1, 2, 3, 4, 5],
    "f6-9 low-moderate cond": [6, 7, 8, 9],
    "f10-14 high-cond unimodal": [10, 11, 12, 13, 14],
    "f15-19 multimodal adequate": [15, 16, 17, 18, 19],
    "f20-24 multimodal weak": [20, 21, 22, 23, 24],
    "f1-24 all noiseless": list(range(1, 25)),
}


def delta_mu_f(precision) -> np.ndarray:
    """Δµf for one or more best-precision values (continuous form)."""
    p = np.asarray(precision, dtype=float)
    p = np.where(np.isfinite(p), p, np.inf)        # unreached -> inf -> Δµf 0
    p = np.maximum(p, TARGET_LO)                    # floor at 1e-13 -> Δµf 1
    val = (LOG_HI - np.log10(p)) / LOG_RANGE
    return np.clip(val, 0.0, 1.0)


def best_precision_at_budget(funvals, budget: float) -> np.ndarray:
    """Best precision reached within ``budget`` true evals, per instance.

    ``funvals``: array with col 0 = eval count, cols 1.. = best precision per
    instance at that eval count (cocopp DataSet.funvals).
    """
    fv = np.asarray(funvals, dtype=float)
    evals = fv[:, 0]
    cols = fv[:, 1:]
    mask = evals <= budget
    n_inst = cols.shape[1]
    out = np.full(n_inst, np.inf)
    if mask.any():
        sub = cols[mask]
        for i in range(n_inst):
            ci = sub[:, i]
            ci = ci[np.isfinite(ci)]
            if ci.size:
                out[i] = float(ci.min())
    return out


def load_variant_datasets(exdata_root, variant: str):
    """cocopp.load the COCO folder for one variant -> DataSetList."""
    import cocopp
    folder = str(Path(exdata_root) / variant)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cocopp.load(folder)


def _dataset_rows(dsl, variant: str, model: str, ec: str, fe_per_d) -> list[dict]:
    """Build long-table rows for one loaded DataSetList (variant or baseline)."""
    rows = []
    for ds in dsl:
        func, dim = int(ds.funcId), int(ds.dim)
        insts = list(ds.instancenumbers)
        fv = np.asarray(ds.funvals, dtype=float)
        for feD in fe_per_d:
            bp = best_precision_at_budget(fv, feD * dim)
            dmf = delta_mu_f(bp)
            for j, inst in enumerate(insts):
                if j >= bp.shape[0]:
                    break
                rows.append({
                    "variant": variant, "model": model, "ec": ec,
                    "function": func, "dimension": dim, "instance": int(inst),
                    "fe_per_d": feD,
                    "best_precision": float(bp[j]),
                    "delta_mu_f": float(dmf[j]),
                })
    return rows


def build_long_table(exdata_root, variants, fe_per_d=(50, 250)) -> pd.DataFrame:
    """Master long table: one row per (variant, function, dim, instance, FE/D checkpoint)."""
    rows = []
    for variant in variants:
        model, ec = parse_variant(variant)
        dsl = load_variant_datasets(exdata_root, variant)
        rows.extend(_dataset_rows(dsl, variant, model, ec, fe_per_d))
    return pd.DataFrame(rows)


def build_baseline_table(baseline_paths: dict, fe_per_d=(50, 250)) -> pd.DataFrame:
    """Long table for archived baselines (model=name, ec='baseline')."""
    import warnings as _w
    import cocopp
    rows = []
    for name, path in baseline_paths.items():
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            dsl = cocopp.load(path)
        rows.extend(_dataset_rows(dsl, name, name, "baseline", fe_per_d))
    return pd.DataFrame(rows)


def group_table(long_df: pd.DataFrame, fe_per_d: int,
                groups: dict[str, list[int]] = NOISELESS_GROUPS) -> pd.DataFrame:
    """Mean Δµf per (variant x function-group) at a given FE/D checkpoint."""
    df = long_df[long_df.fe_per_d == fe_per_d]
    cols = {}
    for gname, funcs in groups.items():
        sub = df[df.function.isin(funcs)]
        cols[gname] = sub.groupby("variant")["delta_mu_f"].mean()
    return pd.DataFrame(cols)


def overall_table(long_df: pd.DataFrame) -> pd.DataFrame:
    """Mean Δµf per variant at each FE/D checkpoint (overall performance)."""
    t = (long_df.groupby(["variant", "fe_per_d"])["delta_mu_f"]
         .mean().unstack("fe_per_d"))
    t.columns = [f"{c} FE/D" for c in t.columns]
    return t.sort_values(t.columns[-1], ascending=False)
