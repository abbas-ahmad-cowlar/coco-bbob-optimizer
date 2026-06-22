"""COCO archived baseline algorithms.

Maps the four protocol baselines to their keys in cocopp's public bbob archive
(``cocopp.archives.bbob``). ``fetch_*`` downloads the data (cached locally by
cocopp) and returns the local path, which ``cocopp.load`` reads like any result
folder — so baselines flow through the same Δµf / convergence machinery as our
variants.

Note: "CMA-ES-2019" is the literal archive entry ``IPOP-CMA-ES-2019_Faury``,
which lives in cocopp's "incomplete" set (it does not cover every BBOB
function/dimension). Comparisons therefore use whatever (function, dimension)
cells it provides; see analyze.py, which intersects on available cells.
"""

from __future__ import annotations

import warnings

# Friendly protocol name -> cocopp bbob archive key.
BASELINES: dict[str, str] = {
    "CMA-ES-2019": "incomplete/2019/IPOP-CMA-ES-2019_Faury.tgz",
    "DTS-CMA-ES": "2017/DTS-CMA-ES_Pitra.tgz",
    "LMM-CMA-ES": "2013/lmm-CMA-ES_auger_noiseless.tgz",
    "LQ-CMA-ES": "2020/lq-CMA-ES_Hansen.tgz",
}


def fetch_baseline(name: str) -> str:
    """Download (if needed) one baseline; return its local path."""
    import cocopp
    if name not in BASELINES:
        raise KeyError(f"unknown baseline '{name}'; choices: {list(BASELINES)}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cocopp.archives.bbob.get(BASELINES[name])


def fetch_baselines(names=None) -> dict[str, str]:
    """Download all requested baselines; return {name: local_path}."""
    names = list(BASELINES) if names is None else list(names)
    return {n: fetch_baseline(n) for n in names}
