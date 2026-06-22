"""
GMM-NP surrogate model (practical implementation).

Implements a Gaussian Mixture Neural predictor.
Returns predictive mean and std from mixture distribution.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class _GMMNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_components: int):
        super().__init__()
        self.n_components = n_components

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(hidden_dim, n_components)
        self.log_std_head = nn.Linear(hidden_dim, n_components)
        self.logits_head = nn.Linear(hidden_dim, n_components)

    def forward(self, x):
        h = self.shared(x)
        means = self.mean_head(h)
        log_stds = self.log_std_head(h)
        logits = self.logits_head(h)

        stds = torch.exp(log_stds).clamp(min=1e-6)
        weights = torch.softmax(logits, dim=-1)

        return means, stds, weights


class GMMNPSurrogate:
    """Gaussian Mixture Neural surrogate for BO."""

    def __init__(
        self,
        n_components: int = 3,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        epochs: int = 500,
        device: str | None = None,
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state

        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model: _GMMNet | None = None
        self._fitted = False
        self._input_dim: int | None = None

        # normalization stats
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean: float = 0.0
        self.y_std: float = 1.0

    def _nll(self, y, means, stds, weights):
        """
        Negative log likelihood for Gaussian mixture.
        """
        y = y.unsqueeze(-1)  # (n, 1)

        normal = torch.distributions.Normal(means, stds)
        log_probs = normal.log_prob(y)

        weighted_log_probs = log_probs + torch.log(weights + 1e-12)
        log_sum = torch.logsumexp(weighted_log_probs, dim=-1)

        return -log_sum.mean()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.atleast_2d(X)
        y = np.asarray(y).ravel()

        n_samples = X.shape[0]
        input_dim = X.shape[1]

        # Use CPU for small datasets to avoid GPU transfer overhead
        train_device = self.device if n_samples >= 500 else torch.device("cpu")

        # Only rebuild model if first call or input dimension changed
        if self.model is None or self._input_dim != input_dim:
            torch.manual_seed(self.random_state or 0)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(self.random_state or 0)
            self._input_dim = input_dim
            self.model = _GMMNet(input_dim, self.hidden_dim, self.n_components).to(train_device)
            train_epochs = self.epochs
        else:
            # Fine-tune from current weights — fewer epochs on incremental data
            train_epochs = max(100, self.epochs // 2)
            self.model.to(train_device)

        # Normalize X and y
        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0) + 1e-8
        self.y_mean = float(y.mean())
        self.y_std = float(y.std()) + 1e-8

        Xn = (X - self.x_mean) / self.x_std
        yn = (y - self.y_mean) / self.y_std

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Validation split (10%) for early stopping with best-weight restore
        val_size = max(1, int(0.1 * n_samples)) if n_samples >= 10 else 0
        rng_np = np.random.RandomState(self.random_state if self.random_state is not None else 0)
        perm = rng_np.permutation(n_samples)

        if val_size > 0 and (n_samples - val_size) >= 5:
            val_idx = perm[:val_size]
            train_idx = perm[val_size:]
        else:
            train_idx = perm
            val_idx = perm

        X_train = torch.tensor(Xn[train_idx], dtype=torch.float32).to(train_device)
        y_train = torch.tensor(yn[train_idx], dtype=torch.float32).to(train_device)
        X_val = torch.tensor(Xn[val_idx], dtype=torch.float32).to(train_device)
        y_val = torch.tensor(yn[val_idx], dtype=torch.float32).to(train_device)

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0
        patience = 15

        self.model.train()
        for _ in range(train_epochs):
            optimizer.zero_grad()
            means, stds, weights = self.model(X_train)
            loss = self._nll(y_train, means, stds, weights)
            loss.backward()
            optimizer.step()

            # Evaluate on validation set
            self.model.eval()
            with torch.no_grad():
                v_means, v_stds, v_weights = self.model(X_val)
                val_loss = self._nll(y_val, v_means, v_stds, v_weights).item()
            self.model.train()

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                no_improve = 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        self._fitted = True

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self._fitted or self.model is None:
            raise RuntimeError("GMMNPSurrogate.predict called before fit().")

        X = np.atleast_2d(X)
        X = (X - self.x_mean) / self.x_std
        predict_device = next(self.model.parameters()).device
        X_tensor = torch.tensor(X, dtype=torch.float32).to(predict_device)

        with torch.no_grad():
            means, stds, weights = self.model(X_tensor)

        means = means.cpu().numpy()
        stds = stds.cpu().numpy()
        weights = weights.cpu().numpy()

        # Predictive mean of mixture
        mean = np.sum(weights * means, axis=1)

        # Predictive variance of mixture:
        # Var = E[σ² + μ²] - (E[μ])²
        second_moment = np.sum(weights * (stds**2 + means**2), axis=1)
        var = second_moment - mean**2
        std = np.sqrt(np.maximum(var, 1e-8))

        # De-normalize
        mean = mean * self.y_std + self.y_mean
        std = std * self.y_std

        return mean.ravel(), std.ravel()

    def run(self, objective, lb, ub, budget=1000, surrogate_interval=5, n_candidates=0,
            random_state=42, ec_type="lq", eval_fraction=0.1,
            eval_fraction_min=0.05, eval_fraction_max=0.5,
            lmm_n_iter_frac=0.05, lmm_tau_good=0.5,
            dts_beta=0.05, dts_epsilon_min=0.05, dts_epsilon_max=0.5,
            lq_tau_threshold=0.85,
            rmse_threshold=0.5, kappa_init=2.0, kappa_final=0.1) -> dict:
        """CMA-ES loop with GMM-NP surrogate and evolution control from Pitra et al. (2021)."""
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