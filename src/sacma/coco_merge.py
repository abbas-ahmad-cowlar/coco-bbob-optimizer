"""Merge per-unit COCO folders into one folder per variant, for cocopp.

The runner writes one COCO folder per (variant, dim, function) unit. cocopp treats
each folder as a separate "algorithm", so to put a variant's full data under one
algorithm we merge its unit folders: append same-named ``.info`` files (each holds
one funcId/DIM block) and copy the ``data_f*`` files (no collisions — every
(func, dim) is unique). Our Δµf/convergence machinery reads the unit folders
directly and does NOT need this; only the cocopp ECDF report does.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def merge_variant(exdata_root, vid: str, dest_root) -> str:
    """Merge all unit folders of one variant into ``dest_root/<vid>``; return its path."""
    exroot = Path(exdata_root)
    dest = Path(dest_root) / vid
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for unit in sorted(exroot.glob(f"{vid}__D*__f*")):
        for info in unit.glob("*.info"):
            content = info.read_text(encoding="utf-8")
            if not content.endswith("\n"):
                content += "\n"
            with (dest / info.name).open("a", encoding="utf-8") as f:
                f.write(content)
        for datadir in unit.glob("data_f*"):
            tgt = dest / datadir.name
            tgt.mkdir(exist_ok=True)
            for df in datadir.iterdir():
                shutil.copy2(df, tgt / df.name)
    return str(dest)


def merge_variants(exdata_root, variants, dest_root) -> dict[str, str]:
    """Merge every variant; return {vid: merged_folder_path}."""
    return {vid: merge_variant(exdata_root, vid, dest_root) for vid in variants}
