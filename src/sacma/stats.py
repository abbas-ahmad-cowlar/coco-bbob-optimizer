"""Pairwise comparison statistics: % wins, Wilcoxon signed-rank + Holm correction.

All comparisons are paired over matched problems (function x dimension x instance)
at a given FE/D checkpoint, using Δµf as the performance measure by default.
"""

from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


def _wide(long_df: pd.DataFrame, fe_per_d: int, metric: str) -> pd.DataFrame:
    """Pivot to problems (rows) x variants (cols) for a checkpoint."""
    df = long_df[long_df.fe_per_d == fe_per_d]
    return df.pivot_table(index=["function", "dimension", "instance"],
                          columns="variant", values=metric)


def pairwise_win_matrix(long_df: pd.DataFrame, fe_per_d: int,
                        metric: str = "delta_mu_f",
                        higher_better: bool = True) -> pd.DataFrame:
    """Percentage of matched problems on which row-variant beats column-variant."""
    w = _wide(long_df, fe_per_d, metric)
    variants = list(w.columns)
    M = pd.DataFrame(np.nan, index=variants, columns=variants)
    for a in variants:
        for b in variants:
            if a == b:
                continue
            pair = w[[a, b]].dropna()
            if len(pair) == 0:
                continue
            wins = (pair[a] > pair[b]).sum() if higher_better else (pair[a] < pair[b]).sum()
            M.loc[a, b] = 100.0 * wins / len(pair)
    return M


def wilcoxon_holm(long_df: pd.DataFrame, fe_per_d: int,
                  metric: str = "delta_mu_f",
                  higher_better: bool = True) -> pd.DataFrame:
    """Two-sided paired Wilcoxon for every variant pair, Holm-corrected."""
    w = _wide(long_df, fe_per_d, metric)
    variants = list(w.columns)
    recs = []
    for a, b in combinations(variants, 2):
        pair = w[[a, b]].dropna()
        n = len(pair)
        mean_a, mean_b = float(pair[a].mean()), float(pair[b].mean())
        better = a if ((mean_a > mean_b) == higher_better) else b
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stat, p = wilcoxon(pair[a].values, pair[b].values)
        except ValueError:  # all-zero differences
            stat, p = np.nan, 1.0
        recs.append({"A": a, "B": b, "n": n, "mean_A": mean_a, "mean_B": mean_b,
                     "better": better, "statistic": stat, "p_raw": p})
    res = pd.DataFrame(recs)
    if len(res):
        rej, pcorr, _, _ = multipletests(res["p_raw"].fillna(1.0).values, method="holm")
        res["p_holm"] = pcorr
        res["significant"] = rej
    return res
