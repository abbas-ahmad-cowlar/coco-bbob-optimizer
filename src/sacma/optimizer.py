"""IPOP-CMA-ES with surrogate prescreening — the protocol-spec base optimizer.

The ``CMAOptimizer`` (models/models/cma_optimizer.py) implements the three
evolution-control (EC) strategies from Pitra et al. (GECCO 2021), but with its own
population/step-size/restart defaults that do NOT match the required protocol. This
module wraps those EC strategies in a proper IPOP restart scheme using the exact
settings from the experiment instructions (the experiment protocol §16):

    Base optimizer  : IPOP-CMA-ES
    Restarts        : 50
    IncPopSize      : 2          (population doubles each restart)
    sigma_start     : 8/3
    Population size : lambda = 8 + floor(6 * ln D)
    Initial point   : x0 ~ U[-4, 4]^D
    Budget          : 250 * D    true black-box evaluations

The EC update rules themselves are reused *verbatim* from the class
(``_update_lmm_ec`` / ``_update_dts_ec`` / ``_update_lq_ec``) so the "proposed
algorithm" behaviour is unchanged — only the CMA-ES base configuration is corrected.

Budget accounting: only TRUE objective() calls count toward the budget. Candidates
assigned surrogate fitness are never evaluated and never counted (matching the
paper's convention that the performance x-axis is true BB evaluations).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import kendalltau, spearmanr

import cma

# Make the package importable (package is literally named ``models``).
_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

from models.cma_optimizer import CMAOptimizer, _compute_normalized_rmse  # noqa: E402

_MAX_ARCHIVE = 300  # surrogate training window (matches project default)


def ipop_lambda(dim: int) -> int:
    """Protocol population size: lambda = 8 + floor(6 * ln D)."""
    return 8 + int(math.floor(6.0 * math.log(dim)))


def default_budget(dim: int) -> int:
    """Protocol budget: 250 * D true evaluations."""
    return 250 * dim


class IPOPSurrogateCMAES:
    """IPOP-CMA-ES + surrogate prescreening with one of the three ECs.

    Parameters
    ----------
    surrogate : object
        Any object with ``fit(X, y)`` and ``predict(X) -> (mean, std)``.
    ec_type : str
        "lmm", "dts", or "lq".
    random_state : int
        Master seed.
    max_restarts : int
        IPOP restart cap (default 50, per protocol).
    inc_popsize : int
        Population multiplier on each restart (default 2).
    sigma0 : float
        Initial step size (default 8/3).
    x0_low, x0_high : float
        Bounds of the uniform initial-point sampling box (default [-4, 4]).
    ec_config : CMAOptimizer | None
        Source of EC hyper-parameters and update methods. If None, a default
        CMAOptimizer is constructed with the given ec_type so the
        defaults (thresholds, eval fractions, RMSE gate, kappa) are reused exactly.
    """

    def __init__(
        self,
        surrogate: Any,
        ec_type: str = "lq",
        random_state: int = 42,
        max_restarts: int = 50,
        inc_popsize: int = 2,
        sigma0: float = 8.0 / 3.0,
        x0_low: float = -4.0,
        x0_high: float = 4.0,
        ec_config: CMAOptimizer | None = None,
    ) -> None:
        if ec_type not in ("lmm", "dts", "lq"):
            raise ValueError(f"ec_type must be lmm/dts/lq, got '{ec_type}'")
        self.surrogate = surrogate
        self.ec_type = ec_type
        self.random_state = random_state
        self.max_restarts = max_restarts
        self.inc_popsize = inc_popsize
        self.sigma0 = sigma0
        self.x0_low = x0_low
        self.x0_high = x0_high
        # Borrow the EC hyper-parameters + update methods (no run() call).
        self.cfg = ec_config or CMAOptimizer(surrogate=surrogate, ec_type=ec_type)

    # ------------------------------------------------------------------
    def run(
        self,
        objective: Callable[[np.ndarray], float],
        lb: np.ndarray,
        ub: np.ndarray,
        budget: int | None = None,
        *,
        is_done: Callable[[], bool] | None = None,
    ) -> dict:
        """Optimise ``objective`` over [lb, ub].

        Parameters
        ----------
        objective : callable
            Maps an x vector to a scalar (minimisation).
        lb, ub : array
            Lower/upper bounds (BBOB domain is [-5, 5]^D).
        budget : int | None
            Total true evaluations. Defaults to 250 * D.
        is_done : callable | None
            Optional early-stop predicate (e.g. COCO ``final_target_hit``).

        Returns
        -------
        dict with: best_x, best_y, n_evals, trace (best-so-far per true eval),
        and diagnostics.
        """
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)
        dim = lb.shape[0]
        budget = int(budget if budget is not None else default_budget(dim))
        cfg = self.cfg

        n_candidates = cfg._n_candidates_cfg or max(200, 100 * dim)
        min_samples = cfg._min_samples_cfg or max(10, 5 * dim)
        rng = np.random.default_rng(self.random_state)

        # ---- Global (persist across restarts) ----------------------------
        X_archive: list[np.ndarray] = []
        y_archive: list[float] = []
        best_x: np.ndarray | None = None
        best_y: float = float("inf")
        n_evals: int = 0
        trace: list[float] = []  # best-so-far f after each true evaluation

        diag = {
            "ec_type": self.ec_type,
            "n_restarts": 0,
            "n_surrogate_generations": 0,
            "n_direct_generations": 0,
            "surrogate_rmse_history": [],
            "n_real_evals_saved": 0,
            "ec_tau_history": [],
            "p_real_history": [],
            "popsize_schedule": [],
        }

        # ---- EC state (persist across restarts) --------------------------
        n_real_lmm = 0  # set per restart relative to popsize
        alpha_dts = float(np.clip(cfg.eval_fraction, cfg.eval_fraction_min, cfg.eval_fraction_max))
        epsilon_dts = 0.0
        k_lq = 0
        tau_queue: list[tuple[float, float]] = []

        def _eval(x: np.ndarray) -> float:
            nonlocal best_x, best_y, n_evals
            f = float(objective(np.asarray(x, dtype=float)))
            xc = np.asarray(x, dtype=float).copy()
            X_archive.append(xc)
            y_archive.append(f)
            if f < best_y:
                best_y = f
                best_x = xc
            n_evals += 1
            trace.append(best_y)
            return f

        popsize = ipop_lambda(dim)

        for restart in range(self.max_restarts + 1):
            if n_evals >= budget or (is_done is not None and is_done()):
                break

            diag["popsize_schedule"].append(popsize)
            x0 = rng.uniform(self.x0_low, self.x0_high, size=dim)
            x0 = np.clip(x0, lb, ub)

            # Per-restart EC counters tied to current popsize.
            n_iter_lmm = max(1, round(popsize * cfg.lmm_n_iter_frac))
            if n_real_lmm == 0:
                n_real_lmm = max(1, round(popsize * cfg.eval_fraction))
            n_real_lmm = int(np.clip(n_real_lmm, 1, popsize))
            if k_lq == 0:
                k_lq = max(1, int(1 + max(0.02 * popsize, 4.0)))
            k_lq = int(np.clip(k_lq, 1, popsize))

            cma_opts = {
                "bounds": [lb.tolist(), ub.tolist()],
                "popsize": popsize,
                "maxfevals": float("inf"),  # budget governed by true-eval guard below
                "seed": int(self.random_state + restart),
                "verbose": -9,
            }
            es = cma.CMAEvolutionStrategy(x0.tolist(), self.sigma0, cma_opts)
            generation = 0

            while not es.stop() and n_evals < budget:
                # No is_done() check here: once the target is hit we let this restart
                # converge deep (~1e-14) so Delta-mu-f is measured below 1e-8. is_done()
                # gates *new restarts* (outer loop) instead — they cannot improve
                # best-so-far once solved, so skipping them saves large amounts of compute.
                progress = min(1.0, n_evals / max(1, budget))
                kappa = cfg.kappa_init + (cfg.kappa_final - cfg.kappa_init) * progress

                use_surrogate = (
                    generation % cfg.surrogate_interval == 0
                    and len(X_archive) >= min_samples
                )

                if use_surrogate:
                    X_fit = np.array(X_archive[-_MAX_ARCHIVE:], dtype=float)
                    y_fit = np.array(y_archive[-_MAX_ARCHIVE:], dtype=float)
                    try:
                        self.surrogate.fit(X_fit, y_fit)
                    except Exception:
                        use_surrogate = False

                if use_surrogate:
                    rmse = _compute_normalized_rmse(self.surrogate, X_fit, y_fit)
                    diag["surrogate_rmse_history"].append(rmse)
                    if rmse >= cfg.rmse_threshold:
                        use_surrogate = False

                surrogate_fitness: dict[int, float] = {}
                sel_mean: np.ndarray | None = None
                n_real_this = popsize

                if use_surrogate:
                    try:
                        pool = es.ask(number=n_candidates)
                        X_cands = np.array(pool, dtype=float)
                        mean_pred, std_pred = self.surrogate.predict(X_cands)
                        scores = -mean_pred + kappa * std_pred
                        top_idx = np.argsort(scores)[-popsize:]
                        solutions = [pool[i] for i in top_idx]
                        sel_mean = mean_pred[top_idx]

                        if self.ec_type == "lmm":
                            n_real_this = int(np.clip(n_real_lmm, 1, popsize))
                        elif self.ec_type == "dts":
                            n_real_this = int(np.clip(round(popsize * alpha_dts), 1, popsize))
                        else:  # lq
                            n_real_this = int(np.clip(k_lq, 1, popsize))

                        for j in range(n_real_this, len(solutions)):
                            surrogate_fitness[j] = float(sel_mean[j])
                            diag["n_real_evals_saved"] += 1

                        diag["p_real_history"].append(n_real_this / popsize)
                        diag["n_surrogate_generations"] += 1
                    except Exception:
                        use_surrogate = False
                        surrogate_fitness = {}
                        sel_mean = None
                        n_real_this = popsize

                if not use_surrogate:
                    solutions = es.ask()
                    diag["n_direct_generations"] += 1

                # ---- Evaluate population -----------------------------------
                fitness: list[float] = []
                real_preds: list[float] = []
                real_true: list[float] = []
                for i, x in enumerate(solutions):
                    if i in surrogate_fitness:
                        fitness.append(surrogate_fitness[i])
                    else:
                        f = _eval(np.array(x))
                        fitness.append(f)
                        if sel_mean is not None and i < len(sel_mean):
                            real_preds.append(float(sel_mean[i]))
                            real_true.append(f)
                    if n_evals >= budget:
                        break

                # ---- EC update (reuse the methods verbatim) -----------
                if use_surrogate and len(real_true) >= 2:
                    if self.ec_type == "lmm":
                        tau, _ = kendalltau(real_preds, real_true)
                        diag["ec_tau_history"].append(0.0 if np.isnan(tau) else float(tau))
                        n_real_lmm = cfg._update_lmm_ec(
                            n_real_lmm, n_iter_lmm, popsize, real_preds, real_true
                        )
                    elif self.ec_type == "dts":
                        rho, _ = spearmanr(real_preds, real_true)
                        diag["ec_tau_history"].append(0.0 if np.isnan(rho) else float(rho))
                        alpha_dts, epsilon_dts = cfg._update_dts_ec(
                            alpha_dts, epsilon_dts, real_preds, real_true
                        )
                    else:  # lq
                        for sp, tv in zip(real_preds, real_true):
                            tau_queue.append((sp, tv))
                        max_q = 20 * popsize
                        if len(tau_queue) > max_q:
                            tau_queue = tau_queue[-max_q:]
                        tau, _ = kendalltau(real_preds, real_true)
                        diag["ec_tau_history"].append(0.0 if np.isnan(tau) else float(tau))
                        k_lq = cfg._update_lq_ec(k_lq, popsize, tau_queue)

                if len(fitness) < popsize // 2 + 1:
                    break
                es.tell(solutions[: len(fitness)], fitness)
                generation += 1

            diag["n_restarts"] = restart + 1
            popsize *= self.inc_popsize  # IPOP: grow population for next restart

        if best_x is None:  # no evaluations happened (shouldn't occur)
            best_x = np.clip(rng.uniform(self.x0_low, self.x0_high, size=dim), lb, ub)

        return {
            "best_x": best_x,
            "best_y": best_y,
            "n_evals": n_evals,
            "trace": np.asarray(trace, dtype=float),
            "diagnostics": diag,
        }
