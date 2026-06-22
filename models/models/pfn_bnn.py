"""
PFN-BNN surrogate model (practical implementation).

Implements uncertainty via deep ensemble.
Provides mean and std similar to GP interface.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class PFNBNNSurrogate:
    """Deep Ensemble surrogate approximating PFN-BNN behaviour."""

    def __init__(
        self,
        n_ensembles: int = 5,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        epochs: int = 500,
        weight_decay: float = 1e-3,
        device: str | None = None,
        random_state: int | None = None,
    ) -> None:
        self.n_ensembles = n_ensembles
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.random_state = random_state
        

        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.models: list[_MLP] = []
        self._fitted = False
        self._input_dim: int | None = None

        # normalization stats
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean: float = 0.0
        self.y_std: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.atleast_2d(X)
        y = np.asarray(y).ravel()

        n_samples = X.shape[0]
        input_dim = X.shape[1]

        # Use CPU for small datasets to avoid GPU transfer overhead
        train_device = self.device if n_samples >= 500 else torch.device("cpu")

        # Only rebuild ensemble if first call or input dimension changed
        if not self.models or self._input_dim != input_dim:
            torch.manual_seed(self.random_state or 0)
            self._input_dim = input_dim
            self.models = [
                _MLP(input_dim, self.hidden_dim).to(train_device)
                for _ in range(self.n_ensembles)
            ]
            train_epochs = self.epochs
        else:
            # Fine-tune from current weights — fewer epochs on incremental data
            train_epochs = max(100, self.epochs // 2)
            for model in self.models:
                model.to(train_device)

        # Normalize X and y
        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0) + 1e-8
        self.y_mean = float(y.mean())
        self.y_std = float(y.std()) + 1e-8

        Xn = (X - self.x_mean) / self.x_std
        yn = (y - self.y_mean) / self.y_std

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
        y_train = torch.tensor(yn[train_idx], dtype=torch.float32).unsqueeze(-1).to(train_device)
        X_val = torch.tensor(Xn[val_idx], dtype=torch.float32).to(train_device)
        y_val = torch.tensor(yn[val_idx], dtype=torch.float32).unsqueeze(-1).to(train_device)

        loss_fn = nn.SmoothL1Loss()

        patience = 15
        for model in self.models:
            optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            model.train()
            best_val_loss = float("inf")
            best_state = None
            no_improve = 0
            for _ in range(train_epochs):
                optimizer.zero_grad()
                preds = model(X_train)
                loss = loss_fn(preds, y_train)
                loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_loss = loss_fn(model(X_val), y_val).item()
                model.train()

                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    no_improve = 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break

            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()

        self._fitted = True

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("PFNBNNSurrogate.predict called before fit().")

        X = np.atleast_2d(X)
        Xn = (X - self.x_mean) / self.x_std
        predict_device = next(self.models[0].parameters()).device
        X_tensor = torch.tensor(Xn, dtype=torch.float32).to(predict_device)

        preds = []
        with torch.no_grad():
            for model in self.models:
                pred = model(X_tensor).cpu().numpy().ravel()
                preds.append(pred)

        preds = np.array(preds)

        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        std = np.maximum(std, 1e-8)

        # De-normalize
        mean = mean * self.y_std + self.y_mean
        std = std * self.y_std

        return mean, std

    def run(self, objective, lb, ub, budget=1000, surrogate_interval=5, n_candidates=0,
            random_state=42, ec_type="lq", eval_fraction=0.1,
            eval_fraction_min=0.05, eval_fraction_max=0.5,
            lmm_n_iter_frac=0.05, lmm_tau_good=0.5,
            dts_beta=0.05, dts_epsilon_min=0.05, dts_epsilon_max=0.5,
            lq_tau_threshold=0.85,
            rmse_threshold=0.5, kappa_init=2.0, kappa_final=0.1) -> dict:
        """CMA-ES loop with PFN-BNN surrogate and evolution control from Pitra et al. (2021)."""
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