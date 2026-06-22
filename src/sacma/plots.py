"""Median convergence curves with quartile bands.

x-axis: true function evaluations per dimension (FE/D, log scale)
y-axis: log10 of best precision (distance to optimum), median over instances,
        with a 25th-75th percentile band.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .metrics import TARGET_HI, TARGET_LO, best_precision_at_budget


def _default_grid(budget_feD: float) -> np.ndarray:
    grid = np.unique(np.concatenate([
        np.geomspace(0.5, budget_feD, 40), [50.0, float(budget_feD)],
    ]))
    return grid[grid <= budget_feD]


@lru_cache(maxsize=None)
def _load(path: str):
    import warnings
    import cocopp
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cocopp.load(str(path))


def convergence_bands(path, function: int, dim: int, feD_grid: np.ndarray):
    """Median / Q1 / Q3 precision over instances at each FE/D grid point."""
    match = [d for d in _load(path) if int(d.funcId) == function and int(d.dim) == dim]
    if not match:
        return None
    fv = np.asarray(match[0].funvals, dtype=float)
    P = np.array([best_precision_at_budget(fv, feD * dim) for feD in feD_grid])
    P = np.where(np.isfinite(P), P, TARGET_HI)
    P = np.maximum(P, TARGET_LO)
    return (np.median(P, axis=1), np.percentile(P, 25, axis=1), np.percentile(P, 75, axis=1))


def plot_convergence(sources: dict, function: int, dim: int, out_path,
                     budget_feD: float = 250, feD_grid=None):
    """One convergence plot for a (function, dimension).

    ``sources``: ordered mapping {label: path}, where path is a COCO result folder
    (our variant) or an archived baseline path. One line+band per source.
    """
    if feD_grid is None:
        feD_grid = _default_grid(budget_feD)
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = 0
    for label, path in sources.items():
        bands = convergence_bands(path, function, dim, feD_grid)
        if bands is None:
            continue
        med, q1, q3 = bands
        line, = ax.plot(feD_grid, np.log10(med), label=label, lw=1.6)
        ax.fill_between(feD_grid, np.log10(q1), np.log10(q3),
                        alpha=0.15, color=line.get_color())
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return None
    ax.set_xscale("log")
    ax.set_xlabel("true function evaluations / dimension")
    ax.set_ylabel(r"$\log_{10}$ best precision $\Delta f$")
    ax.set_title(f"f{function}  D={dim} — median convergence (quartile band)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
