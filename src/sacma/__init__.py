"""sacma — Surrogate-Assisted CMA-ES experiment framework for COCO/BBOB.

Modules (built in phases):
    optimizer  — IPOP-CMA-ES wrapper around the CMAOptimizer (P1)
    runner     — COCO suite driver with cocoex logging + per-eval traces (P2)
    metrics    — Delta-mu-f and the 12 function groups (P3)
    stats      — pairwise win tables, Wilcoxon signed-rank + Holm (P3)
    plots      — median convergence curves with quartiles (P3)
"""

__version__ = "0.1.0"
