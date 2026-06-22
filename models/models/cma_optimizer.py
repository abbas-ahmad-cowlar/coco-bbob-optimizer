"""
CMAOptimizer — CMA-ES with surrogate prescreening and all three evolution controls
from Pitra et al. (GECCO 2021): lmm_EC, DT_EC (DTS-CMA-ES), and lq_EC (lq-CMA-ES).

Reference
---------
Pitra, Hanuš, Koza, Tumpach, Holeňa (2021).
"Interaction between Model and Its Evolution Control in
Surrogate-Assisted CMA Evolution Strategy." GECCO '21.

Three EC strategies (select via ec_type):
------------------------------------------
"lmm"  — lmm-CMA EC (Algorithm 1 in paper)
         n_real starts at 1, adapts via Kendall τ:
           τ ≥ 0.5 (good ranking, i=1): n_real -= n_iter  (save more budget)
           τ < 0.5 (bad  ranking, i≥3): n_real += n_iter  (be conservative)
         n_iter = max(1, round(λ/20))

"dts"  — DTS-CMA-ES EC (Algorithm 2 in paper)
         α starts at eval_fraction (paper default: 0.05).
         ε = exponential moving average of RDE (ranking difference error).
         α = α_min + (α_max - α_min) · clip((ε - ε_min)/(ε_max - ε_min), 0, 1)
         When surrogate ranks well → ε low → α small → few BB evals.
         When surrogate ranks badly → ε high → α large → more BB evals.

"lq"   — lq-CMA-ES EC (Algorithm 3 in paper) — BEST performing EC in paper
         Maintains rolling queue Q of (surrogate_pred, true_val) pairs.
         Kendall τ computed on last L_k = max(15, min(1.2k, 0.75λ)) entries of Q.
         τ < τ_threshold (0.85): k *= 1.5  (expand BB-eval set this generation)
         τ ≥ τ_threshold       : k /= 1.5  (contract — surrogate is reliable)

Comparison note
---------------
  X-axis in ALL performance plots must be number of TRUE black-box evaluations
  (n_evals), never generation count. Surrogate fitness assignments do not call
  objective() and are not counted.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.stats import kendalltau, spearmanr

try:
    import cma
    _CMA_AVAILABLE = True
except ImportError:
    _CMA_AVAILABLE = False


def _compute_normalized_rmse(surrogate: Any, X: np.ndarray, y: np.ndarray) -> float:
    """Normalized in-sample RMSE. Returns RMSE/std(y), or inf on failure."""
    try:
        mean_pred, _ = surrogate.predict(X)
        y_std = float(np.std(y))
        if y_std < 1e-8:
            return 0.0
        return float(np.sqrt(np.mean((mean_pred - y) ** 2)) / y_std)
    except Exception:
        return float("inf")


def _rde_proxy(surr_preds: list[float], true_vals: list[float]) -> float:
    """
    Ranking Difference Error proxy for a single surrogate.
    Uses Spearman rho: RDE_proxy = (1 - max(0, rho)) mapped to [0, 1].
    Perfect ranking (rho=1) → 0.  Random (rho=0) → 0.5.  Reversed (rho=-1) → 1.
    This is the single-surrogate approximation of DTS-CMA-ES's two-GP RDE.
    """
    if len(surr_preds) < 2:
        return 0.5
    try:
        rho, _ = spearmanr(surr_preds, true_vals)
        if np.isnan(rho):
            rho = 0.0
        return float((1.0 - float(rho)) / 2.0)
    except Exception:
        return 0.5


class CMAOptimizer:
    """
    CMA-ES + surrogate prescreening with one of three evolution controls from the paper.

    Parameters
    ----------
    budget : int
        Total real (BB) function evaluations allowed.
    surrogate : object
        Any object with fit(X, y) and predict(X) -> (mean, std).
    surrogate_interval : int
        Apply surrogate prescreening every N CMA-ES generations (default 5).
    n_candidates : int
        Pool size for prescreening. 0 = auto: max(200, 100*dim).
    min_samples : int
        Minimum archive size before surrogate is used. 0 = auto: 5*dim.
    random_state : int
        Master random seed.
    ec_type : str
        Evolution control strategy: "lmm", "dts", or "lq" (default "lq").
        "lq" is the best-performing EC in Pitra et al. (2021).
    eval_fraction : float
        Initial fraction of candidates sent to BB each surrogate generation.
        All three ECs start here and adapt from this value. Default 0.1.
        Paper values: DTS starts at 0.05; lmm and lq start at ~1/λ.
    eval_fraction_min : float
        Hard floor on the BB fraction (default 0.05 ≈ 1 individual for λ≈6-15).
    eval_fraction_max : float
        Hard ceiling on the BB fraction (default 0.5).

    EC-specific parameters
    ----------------------
    lmm_n_iter_frac : float
        Step size for lmm_EC adaptation: n_iter = max(1, round(λ * lmm_n_iter_frac)).
        Paper uses λ/20, so default is 0.05.
    lmm_tau_good : float
        Kendall τ threshold to decide "surrogate was good" (i=1 analogy). Default 0.5.

    dts_beta : float
        Smoothing factor for ε exponential moving average in DT_EC. Default 0.05.
    dts_epsilon_min : float
        ε below this → α = α_min (full trust in surrogate). Default 0.05.
    dts_epsilon_max : float
        ε above this → α = α_max (full mistrust). Default 0.5.

    lq_tau_threshold : float
        Kendall τ threshold for lq_EC. Default 0.85 (from paper).

    rmse_threshold : float
        Quality gate: surrogate skipped if normalized RMSE ≥ this. Default 0.5.
    kappa_init : float
        Initial acquisition exploration weight. Default 2.0.
    kappa_final : float
        Final acquisition exploitation weight. Default 0.1.
    """

    def __init__(
        self,
        budget: int = 1000,
        surrogate: Any = None,
        surrogate_interval: int = 5,
        n_candidates: int = 0,
        min_samples: int = 0,
        random_state: int = 42,
        ec_type: str = "lq",
        eval_fraction: float = 0.1,
        eval_fraction_min: float = 0.05,
        eval_fraction_max: float = 0.5,
        # lmm_EC params
        lmm_n_iter_frac: float = 0.05,
        lmm_tau_good: float = 0.5,
        # DT_EC params
        dts_beta: float = 0.05,
        dts_epsilon_min: float = 0.05,
        dts_epsilon_max: float = 0.5,
        # lq_EC params
        lq_tau_threshold: float = 0.85,
        rmse_threshold: float = 0.5,
        kappa_init: float = 2.0,
        kappa_final: float = 0.1,
    ) -> None:
        if not _CMA_AVAILABLE:
            raise ImportError("cma library required. Install with: pip install cma")
        if surrogate is None:
            raise ValueError("surrogate must be provided.")
        if ec_type not in ("lmm", "dts", "lq"):
            raise ValueError(f"ec_type must be 'lmm', 'dts', or 'lq', got '{ec_type}'")

        self.budget = budget
        self.surrogate = surrogate
        self.surrogate_interval = surrogate_interval
        self._n_candidates_cfg = n_candidates
        self._min_samples_cfg = min_samples
        self.random_state = random_state
        self.ec_type = ec_type
        self.eval_fraction = eval_fraction
        self.eval_fraction_min = eval_fraction_min
        self.eval_fraction_max = eval_fraction_max
        self.lmm_n_iter_frac = lmm_n_iter_frac
        self.lmm_tau_good = lmm_tau_good
        self.dts_beta = dts_beta
        self.dts_epsilon_min = dts_epsilon_min
        self.dts_epsilon_max = dts_epsilon_max
        self.lq_tau_threshold = lq_tau_threshold
        self.rmse_threshold = rmse_threshold
        self.kappa_init = kappa_init
        self.kappa_final = kappa_final

    # ------------------------------------------------------------------
    # EC update methods — one per strategy, called after each surrogate gen
    # ------------------------------------------------------------------

    def _update_lmm_ec(
        self,
        n_real: int,
        n_iter: int,
        popsize: int,
        surr_preds: list[float],
        true_vals: list[float],
    ) -> int:
        """
        lmm_EC update (Algorithm 1, line 19 in paper).
        Returns updated n_real (absolute count, not fraction).

        Good ranking (τ ≥ lmm_tau_good, analogous to i=1):
            n_real -= n_iter  (trust surrogate more next generation)
        Bad  ranking (τ <  lmm_tau_good, analogous to i≥3):
            n_real += n_iter  (be more conservative next generation)
        """
        if len(true_vals) < 2:
            return n_real
        try:
            tau, _ = kendalltau(surr_preds, true_vals)
            if np.isnan(tau):
                tau = 0.0
        except Exception:
            tau = 0.0

        if tau >= self.lmm_tau_good:
            n_real = max(1, n_real - n_iter)
        else:
            n_real = min(popsize, n_real + n_iter)
        return n_real

    def _update_dts_ec(
        self,
        alpha: float,
        epsilon: float,
        surr_preds: list[float],
        true_vals: list[float],
    ) -> tuple[float, float]:
        """
        DT_EC update (Algorithm 2, lines 7-8 in paper).
        Returns (alpha_new, epsilon_new).

        ε = (1 - β)·ε_old + β·RDE_proxy
        α = α_min + (α_max - α_min) · clip((ε - ε_min)/(ε_max - ε_min), 0, 1)

        When RDE is low (surrogate ranks well) → ε low → α small → few BB evals.
        When RDE is high (surrogate ranks badly) → ε high → α large → more BB evals.
        """
        rde = _rde_proxy(surr_preds, true_vals)
        epsilon = (1.0 - self.dts_beta) * epsilon + self.dts_beta * rde
        span = self.dts_epsilon_max - self.dts_epsilon_min
        if span < 1e-10:
            t = 0.0
        else:
            t = float(np.clip((epsilon - self.dts_epsilon_min) / span, 0.0, 1.0))
        alpha = float(np.clip(
            self.eval_fraction_min + (self.eval_fraction_max - self.eval_fraction_min) * t,
            self.eval_fraction_min,
            self.eval_fraction_max,
        ))
        return alpha, epsilon

    def _update_lq_ec(
        self,
        k: int,
        popsize: int,
        tau_queue: list[tuple[float, float]],
    ) -> int:
        """
        lq_EC update (Algorithm 3, lines 9-10 and 19-22 in paper).
        Returns updated k (absolute BB-eval count for next generation).

        L_k = max(15, min(1.2k, 0.75λ))  window size for τ assessment.
        τ < τ_threshold : k *= 1.5  (expand — surrogate not reliable enough)
        τ ≥ τ_threshold : k /= 1.5  (contract — surrogate reliable, save budget)
        """
        L_k = max(15, min(int(1.2 * k), int(0.75 * popsize)))
        recent = tau_queue[-L_k:]
        if len(recent) < 2:
            return k
        sp_vals = [p for p, _ in recent]
        tv_vals = [v for _, v in recent]
        try:
            tau, _ = kendalltau(sp_vals, tv_vals)
            if np.isnan(tau):
                tau = 0.0
        except Exception:
            tau = 0.0

        if tau < self.lq_tau_threshold:
            k = min(popsize, max(1, int(np.ceil(k * 1.5))))
        else:
            k = max(1, int(np.floor(k / 1.5)))
        return k

    def run(
        self,
        objective: Callable[[np.ndarray], float],
        lb: np.ndarray,
        ub: np.ndarray,
    ) -> dict:
        """
        Run CMA-ES + surrogate evolution control.

        Returns
        -------
        dict with keys: X, y, best_y, best_x, n_evals, diagnostics
        """
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)
        dim = lb.shape[0]

        n_candidates = (
            self._n_candidates_cfg if self._n_candidates_cfg > 0
            else max(200, 100 * dim)
        )
        min_samples = (
            self._min_samples_cfg if self._min_samples_cfg > 0
            else max(10, 5 * dim)
        )
        popsize = max(6, 4 + int(3 * np.log(dim)))
        x0 = (lb + ub) / 2.0
        sigma0 = float(np.mean((ub - lb) / 6.0))
        max_archive = 300

        # ---- Initialise EC-specific state --------------------------------
        if self.ec_type == "lmm":
            # n_real_lmm: absolute number of BB evals per surrogate generation
            # starts at eval_fraction * λ (≈1 for default 0.1 and λ≈6-10)
            n_real_lmm = max(1, round(popsize * self.eval_fraction))
            n_iter_lmm = max(1, round(popsize * self.lmm_n_iter_frac))

        elif self.ec_type == "dts":
            # alpha: fraction of population to BB-evaluate (paper starts at 0.05)
            alpha_dts = float(np.clip(
                self.eval_fraction, self.eval_fraction_min, self.eval_fraction_max
            ))
            epsilon_dts = 0.0  # exponential moving average of RDE

        else:  # "lq"
            # k_lq: number of BB evals per surrogate generation (adapts)
            # tau_queue: rolling list of (surrogate_pred, true_val) pairs
            k_lq = max(1, int(1 + max(0.02 * popsize, 4.0)))
            tau_queue: list[tuple[float, float]] = []

        # ---- Archive and tracking ----------------------------------------
        X_archive: list[np.ndarray] = []
        y_archive: list[float] = []
        best_x: np.ndarray | None = None
        best_y: float = float("inf")
        n_evals: int = 0

        n_surrogate_gen: int = 0
        n_direct_gen: int = 0
        surrogate_rmse_history: list[float] = []
        n_real_evals_saved: int = 0
        ec_tau_history: list[float] = []   # Kendall τ (lmm/lq) or Spearman (dts)
        p_real_history: list[float] = []   # fraction sent to BB each surrogate gen

        def _eval(x: np.ndarray) -> float:
            nonlocal best_x, best_y, n_evals
            f = float(objective(np.asarray(x, dtype=float)))
            X_archive.append(np.asarray(x, dtype=float).copy())
            y_archive.append(f)
            if f < best_y:
                best_y = f
                best_x = np.asarray(x, dtype=float).copy()
            n_evals += 1
            return f

        cma_opts = {
            "bounds": [lb.tolist(), ub.tolist()],
            "popsize": popsize,
            "maxfevals": self.budget,
            "seed": self.random_state,
            "verbose": -9,
        }

        es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, cma_opts)
        generation = 0

        while n_evals < self.budget:
            while not es.stop() and n_evals < self.budget:
                solutions = es.ask()

                # Adaptive kappa: linearly decay from kappa_init to kappa_final
                progress = min(1.0, n_evals / max(1, self.budget))
                kappa = self.kappa_init + (self.kappa_final - self.kappa_init) * progress

                use_surrogate = (
                    (generation % self.surrogate_interval == 0)
                    and len(X_archive) >= min_samples
                )

                surrogate_fitness: dict[int, float] = {}
                sel_mean: np.ndarray | None = None

                if use_surrogate:
                    X_fit = np.array(X_archive[-max_archive:], dtype=float)
                    y_fit = np.array(y_archive[-max_archive:], dtype=float)
                    try:
                        self.surrogate.fit(X_fit, y_fit)
                    except Exception:
                        use_surrogate = False

                if use_surrogate:
                    rmse = _compute_normalized_rmse(self.surrogate, X_fit, y_fit)
                    surrogate_rmse_history.append(rmse)
                    if rmse >= self.rmse_threshold:
                        use_surrogate = False

                if use_surrogate:
                    try:
                        large_pool = es.ask(number=n_candidates)
                        X_cands = np.array(large_pool, dtype=float)
                        mean_pred, std_pred = self.surrogate.predict(X_cands)
                        scores = -mean_pred + kappa * std_pred
                        top_idx = np.argsort(scores)[-popsize:]
                        solutions = [large_pool[i] for i in top_idx]
                        sel_mean = mean_pred[top_idx]

                        # ---- Determine n_real for this generation -----------
                        if self.ec_type == "lmm":
                            n_real_this = int(np.clip(n_real_lmm, 1, popsize))
                        elif self.ec_type == "dts":
                            n_real_this = int(np.clip(
                                max(1, round(popsize * alpha_dts)), 1, popsize
                            ))
                        else:  # lq
                            n_real_this = int(np.clip(k_lq, 1, popsize))

                        # Candidates beyond n_real_this use surrogate fitness
                        for sol_idx in range(n_real_this, len(solutions)):
                            surrogate_fitness[sol_idx] = float(sel_mean[sol_idx])
                            n_real_evals_saved += 1

                        p_real_history.append(n_real_this / popsize)
                        n_surrogate_gen += 1
                    except Exception:
                        use_surrogate = False
                        n_real_this = popsize

                if not use_surrogate:
                    n_direct_gen += 1
                    n_real_this = popsize

                # ---- Evaluate population --------------------------------
                fitness_values: list[float] = []
                real_surr_preds: list[float] = []
                real_true_vals: list[float] = []

                for i, x in enumerate(solutions):
                    if i in surrogate_fitness:
                        fitness_values.append(surrogate_fitness[i])
                    else:
                        f = _eval(np.array(x))
                        fitness_values.append(f)
                        if use_surrogate and sel_mean is not None and i < len(sel_mean):
                            real_surr_preds.append(float(sel_mean[i]))
                            real_true_vals.append(f)
                    if n_evals >= self.budget:
                        break

                # ---- Update EC state after this generation ---------------
                if use_surrogate and len(real_true_vals) >= 2:
                    if self.ec_type == "lmm":
                        tau, _ = kendalltau(real_surr_preds, real_true_vals)
                        tau = 0.0 if np.isnan(tau) else float(tau)
                        ec_tau_history.append(tau)
                        n_real_lmm = self._update_lmm_ec(
                            n_real_lmm, n_iter_lmm, popsize,
                            real_surr_preds, real_true_vals
                        )

                    elif self.ec_type == "dts":
                        rho, _ = spearmanr(real_surr_preds, real_true_vals)
                        rho = 0.0 if np.isnan(rho) else float(rho)
                        ec_tau_history.append(rho)
                        alpha_dts, epsilon_dts = self._update_dts_ec(
                            alpha_dts, epsilon_dts,
                            real_surr_preds, real_true_vals
                        )

                    else:  # lq
                        for sp, tv in zip(real_surr_preds, real_true_vals):
                            tau_queue.append((sp, tv))
                        # Keep queue bounded (max 20×λ pairs)
                        max_q = 20 * popsize
                        if len(tau_queue) > max_q:
                            tau_queue = tau_queue[-max_q:]
                        tau, _ = kendalltau(real_surr_preds, real_true_vals)
                        tau = 0.0 if np.isnan(tau) else float(tau)
                        ec_tau_history.append(tau)
                        k_lq = self._update_lq_ec(k_lq, popsize, tau_queue)

                if len(fitness_values) < popsize // 2 + 1:
                    break

                es.tell(solutions[: len(fitness_values)], fitness_values)
                generation += 1

            if n_evals < self.budget:
                restart_sigma = min(sigma0 * 2.0, float(np.mean(ub - lb)) / 4.0)
                if best_x is not None and np.random.rand() < 0.5:
                    perturbation = np.random.randn(dim) * restart_sigma * 0.5
                    new_x0 = np.clip(best_x + perturbation, lb, ub)
                else:
                    new_x0 = np.random.uniform(lb, ub)
                cma_opts["seed"] = self.random_state + n_evals
                es = cma.CMAEvolutionStrategy(new_x0.tolist(), restart_sigma, cma_opts)

        X_arr = np.array(X_archive, dtype=float)
        y_arr = np.array(y_archive, dtype=float)

        # Compute final p_real for diagnostics
        if self.ec_type == "lmm":
            p_real_final = n_real_lmm / popsize
        elif self.ec_type == "dts":
            p_real_final = alpha_dts
        else:
            p_real_final = k_lq / popsize

        return {
            "X": X_arr,
            "y": y_arr,
            "best_y": best_y,
            "best_x": best_x if best_x is not None else X_arr[int(np.argmin(y_arr))],
            "n_evals": n_evals,
            "diagnostics": {
                "ec_type": self.ec_type,
                "n_surrogate_generations": n_surrogate_gen,
                "n_direct_generations": n_direct_gen,
                "surrogate_rmse_history": surrogate_rmse_history,
                "n_real_evals_saved": n_real_evals_saved,
                "ec_tau_history": ec_tau_history,
                "p_real_history": p_real_history,
                "p_real_final": p_real_final,
            },
        }
