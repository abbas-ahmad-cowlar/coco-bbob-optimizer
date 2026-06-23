#!/usr/bin/env python
"""Clean, report-ready figures from results/processed/*.csv (+ raw COCO for curves).

Writes to results/processed/report_figs/:
  fig_overall_ranking.png   - Delta-mu-f @250 FE/D, all variants + baselines, sorted
  fig_model_means.png       - mean Delta-mu-f per surrogate model (@50 and @250)
  fig_ec_means.png          - mean Delta-mu-f per evolution control
  fig_group_heatmap.png     - Delta-mu-f per (algorithm x 6 function groups)
  fig_ablation.png          - surrogate RMSE vs true-evals saved (the quality-gate story)
  fig_convergence.png       - our 2 best vs 3 baselines on f1/f10/f15/f20 at D5
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
from sacma.baselines import fetch_baselines  # noqa: E402
from sacma.plots import _default_grid, convergence_bands  # noqa: E402

PROC = _REPO / "results" / "processed"
OUT = PROC / "report_figs"
OUT.mkdir(parents=True, exist_ok=True)
BASELINES = ["CMA-ES-2019", "DTS-CMA-ES", "LMM-CMA-ES", "LQ-CMA-ES"]
OURS_C, BASE_C = "#2c7fb8", "#d95f0e"

overall = pd.read_csv(PROC / "overall_table.csv", index_col=0)
groups = pd.read_csv(PROC / "groups_250FEbyD.csv", index_col=0)
diag = pd.read_csv(PROC / "diagnostics_summary.csv")
long = pd.read_csv(PROC / "long_table.csv")
ours = long[long.ec != "baseline"]


# 1 — overall ranking ------------------------------------------------------
o = overall.sort_values("250 FE/D")
colors = [BASE_C if v in BASELINES else OURS_C for v in o.index]
fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(range(len(o)), o["250 FE/D"], color=colors)
ax.set_yticks(range(len(o)))
ax.set_yticklabels(o.index, fontsize=8)
ax.set_xlabel("mean $\\Delta\\mu_f$ @ 250 FE/D  (higher = better)")
ax.set_title("Overall ranking — surrogate-assisted variants vs published baselines (D2,3,5)")
ax.axvline(o.loc[[b for b in BASELINES if b in o.index], "250 FE/D"].min(), ls=":", c="grey", lw=1)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=OURS_C, label="our variants"),
                   Patch(color=BASE_C, label="baselines")], loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig_overall_ranking.png", dpi=130); plt.close(fig)


# 2 — model means ----------------------------------------------------------
mm = (ours.groupby(["model", "fe_per_d"])["delta_mu_f"].mean().unstack("fe_per_d"))
mm = mm.sort_values(250, ascending=False)
fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(mm))
ax.bar(x - 0.2, mm[50], 0.4, label="50 FE/D", color="#a6cee3")
ax.bar(x + 0.2, mm[250], 0.4, label="250 FE/D", color=OURS_C)
ax.set_xticks(x); ax.set_xticklabels(mm.index, rotation=20, ha="right", fontsize=8)
ax.set_ylabel("mean $\\Delta\\mu_f$"); ax.set_title("Surrogate model — averaged over ECs / functions / dims")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "fig_model_means.png", dpi=130); plt.close(fig)


# 3 — EC means -------------------------------------------------------------
ec = ours[ours.fe_per_d == 250].groupby("ec")["delta_mu_f"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(ec.index, ec.values, color=OURS_C, width=0.55)
ax.set_ylim(ec.min() * 0.98, ec.max() * 1.02)
ax.set_ylabel("mean $\\Delta\\mu_f$ @250 FE/D"); ax.set_title("Evolution control")
for i, v in enumerate(ec.values):
    ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig_ec_means.png", dpi=130); plt.close(fig)


# 4 — group heatmap --------------------------------------------------------
order = groups["f1-24 all noiseless"].sort_values(ascending=False).index
gh = groups.loc[order]
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(gh.values, aspect="auto", cmap="viridis")
ax.set_xticks(range(len(gh.columns))); ax.set_xticklabels(gh.columns, rotation=30, ha="right", fontsize=8)
ax.set_yticks(range(len(gh.index))); ax.set_yticklabels(gh.index, fontsize=7)
for i in range(gh.shape[0]):
    for j in range(gh.shape[1]):
        v = gh.values[i, j]
        if np.isfinite(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if v < 0.55 else "black")
fig.colorbar(im, ax=ax, label="$\\Delta\\mu_f$ @250 FE/D")
ax.set_title("Δµf by function group (baselines + variants)")
fig.tight_layout(); fig.savefig(OUT / "fig_group_heatmap.png", dpi=130); plt.close(fig)


# 5 — ablation: RMSE vs evals saved ---------------------------------------
diag["model"] = diag["variant"].str.replace(r"_(lmm|dts|lq)EC$", "", regex=True)
fig, ax = plt.subplots(figsize=(7.5, 5))
models = list(diag.model.unique())
cmap = plt.cm.tab10
for k, m in enumerate(models):
    d = diag[diag.model == m]
    ax.scatter(d.mean_rmse, d.mean_evals_saved, s=70, color=cmap(k), label=m, edgecolor="k", lw=0.4)
ax.axvline(0.5, ls="--", c="red", lw=1)
ax.text(0.5, ax.get_ylim()[1] * 0.95, " RMSE quality gate (0.5)", color="red", fontsize=8, va="top")
ax.set_xlabel("mean normalized surrogate RMSE  (lower = better surrogate)")
ax.set_ylabel("mean true evaluations saved / run")
ax.set_title("Why GP wins: surrogate accuracy vs evaluations saved")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "fig_ablation.png", dpi=130); plt.close(fig)


# 6 — convergence: our 2 best vs 3 baselines on 4 functions at D5 ----------
bl = fetch_baselines(["DTS-CMA-ES", "LMM-CMA-ES", "LQ-CMA-ES"])
sources = {  # label -> COCO folder path
    "gp_dtsEC": str(_REPO / "exdata" / "gp_dtsEC__D5__f{f}"),
    "pfn_transformer_lqEC": str(_REPO / "exdata" / "pfn_transformer_lqEC__D5__f{f}"),
    **bl,
}
colors = {"gp_dtsEC": OURS_C, "pfn_transformer_lqEC": "#1b9e77",
          "DTS-CMA-ES": "#d95f0e", "LMM-CMA-ES": "#7570b3", "LQ-CMA-ES": "#e7298a"}
grid = _default_grid(250)
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
titles = {1: "f1 Sphere", 10: "f10 Ellipsoid", 15: "f15 Rastrigin", 20: "f20 Schwefel"}
for ax, func in zip(axes.ravel(), [1, 10, 15, 20]):
    for label, path in sources.items():
        p = path.format(f=func) if "{f}" in path else path
        bands = convergence_bands(p, func, 5, grid)
        if bands is None:
            continue
        med, q1, q3 = bands
        ax.plot(grid, np.log10(med), label=label, color=colors[label], lw=1.7)
        ax.fill_between(grid, np.log10(q1), np.log10(q3), color=colors[label], alpha=0.12)
    ax.set_xscale("log"); ax.set_title(f"{titles[func]}  (D5)", fontsize=10)
    ax.set_xlabel("FE / dimension"); ax.set_ylabel("log10 $\\Delta f$"); ax.grid(alpha=0.3, which="both")
axes[0, 0].legend(fontsize=7, loc="upper right")
fig.suptitle("Median convergence (quartile band): our two best vs baselines", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "fig_convergence.png", dpi=130); plt.close(fig)

print("wrote figures to", OUT)
for p in sorted(OUT.glob("*.png")):
    print("  ", p.name, p.stat().st_size, "bytes")
