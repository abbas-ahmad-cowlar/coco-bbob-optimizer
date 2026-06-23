#!/usr/bin/env python
"""Generate the standard cocopp ECDF report (HTML + PNG) for all variants + baselines.

Merges the per-unit COCO folders into one folder per variant, fetches the four
archived baselines, and runs cocopp.main -> ppdata/. Skips rebuilding the Delta-mu-f
tables (those are produced by analyze.py).
"""
from __future__ import annotations

import glob
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import cocopp  # noqa: E402

from sacma.baselines import fetch_baselines  # noqa: E402
from sacma.coco_merge import merge_variants  # noqa: E402
from sacma.runner import parse_unit  # noqa: E402

diag = _REPO / "results" / "diagnostics"
vids = sorted({parse_unit(Path(p).stem)[0] for p in glob.glob(str(diag / "*.done"))})
print(f"variants: {len(vids)} -> {vids}")

merged = merge_variants(_REPO / "exdata", vids, _REPO / "results" / "processed" / "merged_coco")
print(f"merged {len(merged)} variant folders")

baselines = fetch_baselines()
print(f"baselines: {list(baselines)}")

argstr = list(merged.values()) + list(baselines.values())
print(f"running cocopp.main on {len(argstr)} datasets ...")
cocopp.main(" ".join(argstr))
print("ECDF report written under ppdata/")
