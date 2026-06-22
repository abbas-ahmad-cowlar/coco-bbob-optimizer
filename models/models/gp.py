"""
Gaussian Process surrogate model for Bayesian Optimization.

Uses scikit-learn's GaussianProcessRegressor with an ARD Matern + WhiteKernel.
Exposes fit(X, y) and predict(X) -> (mean, std) for the BO loop.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


class GPSurrogate:
    """GP surrogate with fit / predict interface for BO."""

    def __init__(
        self,
        kernel=None,
        alpha: float = 1e-6,
        normalize_y: bool = True,
        random_state: int | None = None,
        n_restarts_optimizer: int = 3,
    ) -> None:
        self._kernel = kernel
        self._alpha = alpha
        self._normalize_y = normalize_y
        self._random_state = random_state
        self._n_restarts_optimizer = int(n_restarts_optimizer)

        self.gpr: GaussianProcessRegressor | None = None
        self._fitted = False
        self._input_dim: int | None = None

    def _build_default_kernel(self, dim: int):
        # ARD Matern 2.5 + WhiteKernel for noise modelling.
        # ConstantKernel allows the GP signal variance to adapt to function scale.
        matern = Matern(
            length_scale=np.ones(dim),
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5,
        )
        white = WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
        return ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * matern + white

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the GP on (X, y). y can be 1d (n,) or (n, 1)."""
        X = np.atleast_2d(X)
        y = np.asarray(y).ravel()
        dim = int(X.shape[1])

        # Rebuild if first call or input dimension changed
        if self.gpr is None or self._input_dim != dim:
            self._input_dim = dim
            kernel = self._kernel if self._kernel is not None else self._build_default_kernel(dim)
            self.gpr = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self._alpha,
                normalize_y=self._normalize_y,
                random_state=self._random_state,
                n_restarts_optimizer=self._n_restarts_optimizer,
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.gpr.fit(X, y)
        self._fitted = True

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std). Std is clipped from below for numerical stability."""
        if not self._fitted or self.gpr is None:
            raise RuntimeError("GPSurrogate.predict called before fit().")
        X = np.atleast_2d(X)
        mean, std = self.gpr.predict(X, return_std=True)
        std = np.maximum(std, 1e-8)
        return mean.ravel(), std.ravel()

    def run(self, objective, lb, ub, budget=1000, surrogate_interval=5, n_candidates=0,
            random_state=42, ec_type="lq", eval_fraction=0.1,
            eval_fraction_min=0.05, eval_fraction_max=0.5,
            lmm_n_iter_frac=0.05, lmm_tau_good=0.5,
            dts_beta=0.05, dts_epsilon_min=0.05, dts_epsilon_max=0.5,
            lq_tau_threshold=0.85,
            rmse_threshold=0.5, kappa_init=2.0, kappa_final=0.1) -> dict:
        """CMA-ES loop with GP surrogate and evolution control from Pitra et al. (2021)."""
        from models.cma_optimizer import CMAOptimizer
        return CMAOptimizer(
            budget=budget, surrogate=self, surrogate_interval=surrogate_interval,
            n_candidates=n_candidates, random_state=random_state,
            ec_type=ec_type, eval_fraction=eval_fraction,
            eval_fraction_min=eval_fraction_min, eval_fraction_max=eval_fraction_max,
            lmm_n_iter_frac=lmm_n_iter_frac, lmm_tau_good=lmm_tau_good,
            dts_beta=dts_beta, dts_epsilon_min=dts_epsilon_min, dts_epsilon_max=dts_epsilon_max,
            lq_tau_threshold=lq_tau_threshold,
            rmse_threshold=rmse_threshold, kappa_init=kappa_init, kappa_final=kappa_final,
        ).run(objective, lb, ub)
