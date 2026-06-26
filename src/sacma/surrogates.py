"""Registry for the six surrogate models in the models package.

The surrogate models live in ``<repo>/models/models/`` (a package literally named
``models``). We add ``<repo>/models`` to ``sys.path`` so ``import models.gp`` etc.
resolve, without modifying those files.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
_MODELS_DIR = _REPO / "models"
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

from models.bnn_mc_dropout import BNNMCDropoutSurrogate  # noqa: E402
from models.gmm_np import GMMNPSurrogate  # noqa: E402
from models.gp import GPSurrogate  # noqa: E402
from models.pfn_bnn import PFNBNNSurrogate  # noqa: E402
from models.pfn_hebo import PFNHEBOSurrogate  # noqa: E402
from models.pfn_transformer import PFNTransformerSurrogate  # noqa: E402

# Canonical model name -> class. Order matches the experiment protocol.
SURROGATES: dict[str, type] = {
    "gp": GPSurrogate,
    "bnn_mc_dropout": BNNMCDropoutSurrogate,
    "pfn_bnn": PFNBNNSurrogate,
    "pfn_hebo": PFNHEBOSurrogate,
    "gmm_np": GMMNPSurrogate,
    "pfn_transformer": PFNTransformerSurrogate,
}

# Three evolution-control strategies (Pitra et al., GECCO 2021).
EC_TYPES: tuple[str, ...] = ("lmm", "dts", "lq")


def make_surrogate(name: str, random_state: int | None = None, **kwargs: Any):
    """Instantiate a surrogate by canonical name.

    All six wrappers accept ``random_state``; it is forwarded for reproducibility.
    """
    if name not in SURROGATES:
        raise KeyError(f"unknown surrogate '{name}'; choices: {list(SURROGATES)}")
    return SURROGATES[name](random_state=random_state, **kwargs)


def variant_id(model: str, ec: str) -> str:
    """Algorithm-variant label used in COCO output, e.g. 'pfn_transformer_lqEC'."""
    return f"{model}_{ec}EC"


def parse_variant(vid: str) -> tuple[str, str]:
    """Inverse of variant_id: 'pfn_transformer_lqEC' -> ('pfn_transformer', 'lq')."""
    for ec in EC_TYPES:
        suffix = f"{ec}EC"
        if vid.endswith(suffix):
            return vid[: -len(suffix)].rstrip("_"), ec
    raise ValueError(f"cannot parse variant id '{vid}'")
