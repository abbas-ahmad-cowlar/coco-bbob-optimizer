"""
Bayesian Neural Network with MC Dropout surrogate model for Bayesian Optimization.

Uses PyTorch to implement a feedforward network with dropout layers.
At prediction time, performs Monte Carlo sampling (multiple forward passes with
dropout enabled) to estimate predictive mean and uncertainty.

Architecture and training follow the AFN (coco-afn-surrogate) reference:
- Pyramid network: 64 -> 32 -> 16 -> 1
- Validation split (10%) with best-weight restore
- Full-batch training (stable for small BO datasets)
- LR=0.001, weight_decay=0.001

Exposes fit(X, y) and predict(X) -> (mean, std) for the BO loop.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class BNNMCDropoutSurrogate:
    """
    Bayesian Neural Network with MC Dropout surrogate.

    Interface matches GPSurrogate: fit(X, y) and predict(X) -> (mean, std).
    """

    def __init__(
        self,
        hidden_sizes: list[int] | None = None,
        dropout_rate: float | None = None,
        learning_rate: float = 0.001,
        n_epochs: int = 300,
        batch_size: int | None = None,
        weight_decay: float = 0.001,
        normalize_y: bool = True,
        n_mc_samples: int | None = None,
        random_state: int | None = None,
        device: str | None = None,
        n_ensemble: int = 1,
    ) -> None:
        """
        Initialize BNN-MC Dropout surrogate.

        Parameters
        ----------
        hidden_sizes : list[int] | None
            Hidden layer sizes. If None (default), adapts automatically:
            hidden_size = min(128, max(32, 8 * input_dim)),
            architecture = input → h → h → h//2 → 1.
        dropout_rate : float | None
            Dropout probability. If None (default), adapts automatically:
            dropout = min(0.4, 0.1 + 0.01 * input_dim).
            Low dim → less noise; high dim → more uncertainty spread.
        learning_rate : float
            Adam optimizer learning rate.
        n_epochs : int
            Maximum training epochs.
        batch_size : int | None
            Batch size. If None, uses full-batch (all samples at once).
        weight_decay : float
            L2 regularization strength.
        normalize_y : bool
            Whether to normalize y targets (zero mean, unit variance).
        n_mc_samples : int | None
            MC passes for uncertainty estimation. If None (default), adapts:
            n_mc = 50 + 2 * input_dim.
        random_state : int | None
            Random seed for reproducibility.
        device : str | None
            PyTorch device string (e.g. 'cuda', 'cpu', 'mps'). If None, auto-detects.
        """
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = float(dropout_rate) if dropout_rate is not None else None
        self.learning_rate = float(learning_rate)
        self.n_epochs = int(n_epochs)
        self.batch_size = batch_size
        self.weight_decay = float(weight_decay)
        self.normalize_y = bool(normalize_y)
        self.n_mc_samples = int(n_mc_samples) if n_mc_samples is not None else None
        self.n_ensemble = max(1, int(n_ensemble))
        self.random_state = random_state
        self._fit_count: int = 0

        if device is not None:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

        self.net: nn.Module | None = None
        self.nets: list[nn.Module] = []  # all ensemble members (includes self.net)
        self._fitted = False
        self._input_dim: int | None = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._temp_scale: float = 1.0  # temperature scaling for uncertainty calibration

        if random_state is not None:
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_state)
            elif torch.backends.mps.is_available():
                torch.mps.manual_seed(random_state)

    def _build_network(self, input_dim: int) -> nn.Module:
        """Build an adaptive feedforward network with dropout layers.

        When hidden_sizes is None (default), capacity scales with input_dim:
          hidden_size = min(128, max(32, 8 * input_dim))
          architecture: input → h → h → h//2 → 1

        When dropout_rate is None (default), it scales with input_dim:
          dropout = min(0.4, 0.1 + 0.01 * input_dim)
        """
        if self.hidden_sizes is None:
            # Adaptive capacity: small for low-dim (avoids overfit), grows for high-dim
            h = min(128, max(32, 8 * input_dim))
            hidden_sizes = [h, h, h // 2]
        else:
            hidden_sizes = list(self.hidden_sizes)

        # Adaptive dropout: low dim needs less noise; high dim needs more spread
        if self.dropout_rate is None:
            dropout = min(0.4, 0.1 + 0.01 * input_dim)
        else:
            dropout = self.dropout_rate

        layers = []
        dim_in = input_dim
        for dim_out in hidden_sizes:
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            dim_in = dim_out

        layers.append(nn.Linear(dim_in, 1))
        return nn.Sequential(*layers)

    def _fit_one(
        self,
        net: nn.Module,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        train_epochs: int,
        bs: int,
    ) -> nn.Module:
        """Train one network with early stopping + best-weight restore. Returns trained net."""
        n_train = X_train.shape[0]
        optimizer = optim.Adam(net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.SmoothL1Loss()
        best_val_loss = float("inf")
        best_state = None
        no_improve = 0
        patience = 15

        net.train()
        for _ in range(train_epochs):
            if bs >= n_train:
                optimizer.zero_grad()
                criterion(net(X_train), y_train).backward()
                optimizer.step()
            else:
                for start in range(0, n_train, bs):
                    batch_idx = torch.randperm(n_train)[start:start + bs]
                    optimizer.zero_grad()
                    criterion(net(X_train[batch_idx]), y_train[batch_idx]).backward()
                    optimizer.step()

            # Validate with dropout ON (train mode) using MC passes —
            # same mode as predict(), so checkpoint selection is aligned with inference.
            n_val_mc = 10
            with torch.no_grad():
                X_val_rep = X_val.unsqueeze(0).expand(n_val_mc, -1, -1).reshape(
                    n_val_mc * X_val.shape[0], -1
                )
                val_preds = net(X_val_rep).reshape(n_val_mc, X_val.shape[0]).mean(dim=0, keepdim=True).T
                val_loss = criterion(val_preds, y_val).item()

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                no_improve = 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        return net

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the BNN on (X, y). y can be 1d (n,) or (n, 1).

        Follows AFN training protocol:
        - 10% validation split for early stopping
        - Best model weights restored before returning
        - Full-batch gradient steps (stable for small BO datasets)
        - Rebuild every 50 calls to avoid weight drift from shifting normalization
        """
        X = np.atleast_2d(X)
        y = np.asarray(y).ravel()

        n_samples, input_dim = X.shape

        # CPU for small datasets (avoids GPU transfer overhead, as in AFN)
        train_device = self.device if n_samples >= 500 else torch.device("cpu")

        # Z-score normalize X
        self._x_mean = X.mean(axis=0)
        self._x_std = np.where(X.std(axis=0) < 1e-8, 1.0, X.std(axis=0))
        X_norm = (X - self._x_mean) / self._x_std

        # Z-score normalize y
        if self.normalize_y:
            self._y_mean = float(np.mean(y))
            self._y_std = float(np.std(y))
            if self._y_std < 1e-10:
                self._y_std = 1.0
            y_norm = (y - self._y_mean) / self._y_std
        else:
            y_norm = y.copy()

        # Rebuild check: use stored _input_dim BEFORE updating it
        rebuild = (
            self.net is None
            or self._input_dim != input_dim
            or self._fit_count % 50 == 0
        )
        self._input_dim = input_dim  # update AFTER the rebuild check

        if rebuild:
            self.net = self._build_network(input_dim)
            train_epochs = self.n_epochs
        else:
            # Fine-tune: use half of n_epochs, but never more than n_epochs itself
            train_epochs = max(1, self.n_epochs // 2)
        self._fit_count += 1
        self.net = self.net.to(train_device)

        # Validation split (10%, like AFN) — used for early stopping and best-weight restore
        val_size = max(1, int(0.1 * n_samples)) if n_samples >= 10 else 0
        rng_np = np.random.RandomState(self.random_state if self.random_state is not None else 0)
        perm = rng_np.permutation(n_samples)

        if val_size > 0 and (n_samples - val_size) >= 5:
            val_idx = perm[:val_size]
            train_idx = perm[val_size:]
        else:
            # Too few samples for a split — train on all, validate on all
            train_idx = perm
            val_idx = perm

        X_train = torch.tensor(X_norm[train_idx], dtype=torch.float32, device=train_device)
        y_train = torch.tensor(y_norm[train_idx], dtype=torch.float32, device=train_device).reshape(-1, 1)
        X_val = torch.tensor(X_norm[val_idx], dtype=torch.float32, device=train_device)
        y_val = torch.tensor(y_norm[val_idx], dtype=torch.float32, device=train_device).reshape(-1, 1)

        # Batch size: full-batch by default (AFN trains full-batch; stable for <300 pts)
        if self.batch_size is not None:
            bs = min(self.batch_size, len(train_idx))
        else:
            bs = len(train_idx)  # full batch

        n_train = len(train_idx)
        bs = min(bs, n_train)

        # Train primary network
        self.net = self._fit_one(self.net, X_train, y_train, X_val, y_val, train_epochs, bs)

        # Temperature scaling: calibrate MC Dropout uncertainty on validation data.
        # Computes T = sqrt(mean((residual/std)^2)) in normalised y-space.
        # T > 1 → model is overconfident (std too small) → inflate std.
        # T < 1 → model is underconfident → deflate std.
        self._temp_scale = 1.0
        if val_size > 0 and len(val_idx) >= 3:
            n_temp_mc = 30
            self.net.train()  # keep dropout active for MC passes
            with torch.no_grad():
                X_val_rep = X_val.unsqueeze(0).expand(n_temp_mc, -1, -1).reshape(
                    n_temp_mc * len(val_idx), -1
                )
                temp_preds = self.net(X_val_rep).reshape(n_temp_mc, len(val_idx)).cpu().numpy()
            val_mean = temp_preds.mean(axis=0)
            val_std = np.maximum(temp_preds.std(axis=0, ddof=0), 1e-6)
            y_val_np = y_norm[val_idx]
            calib = float(np.mean(((y_val_np - val_mean) / val_std) ** 2))
            self._temp_scale = float(np.clip(np.sqrt(max(calib, 1e-8)), 0.3, 10.0))

        self.net.eval()

        # Ensemble: train additional members on bootstrap samples for epistemic diversity.
        # Each member sees a different random subset → members disagree near optimum
        # → combined std stays high even where a single net would collapse.
        self.nets = [self.net]
        if self.n_ensemble > 1:
            for _ in range(1, self.n_ensemble):
                boot_idx = rng_np.choice(n_train, size=n_train, replace=True)
                X_boot = X_train[boot_idx]
                y_boot = y_train[boot_idx]
                net_i = self._build_network(input_dim).to(train_device)
                net_i = self._fit_one(net_i, X_boot, y_boot, X_val, y_val, train_epochs, bs)
                self.nets.append(net_i)

        self._fitted = True

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and uncertainty using MC Dropout sampling.

        Parameters
        ----------
        X : np.ndarray
            Test inputs, shape (n_samples, n_features).

        Returns
        -------
        mean : np.ndarray
            Predictive mean, shape (n_samples,).
        std : np.ndarray
            Predictive standard deviation, shape (n_samples,).
        """
        if not self._fitted or self.net is None:
            raise RuntimeError("BNNMCDropoutSurrogate.predict called before fit().")

        # Adaptive MC samples: more passes needed in higher dims for stable std estimate
        n_mc = (
            self.n_mc_samples
            if self.n_mc_samples is not None
            else 50 + 2 * (self._input_dim or 1)
        )

        if n_mc < 2:
            warnings.warn(
                "MC Dropout uncertainty estimates require at least 2 samples; "
                f"consider increasing n_mc_samples (currently {n_mc}).",
                UserWarning,
            )

        X = np.atleast_2d(X)
        X = (X - self._x_mean) / self._x_std
        predict_device = next(self.net.parameters()).device
        X_tensor = torch.tensor(X, dtype=torch.float32, device=predict_device)

        n_test = X_tensor.shape[0]
        # Pool predictions from all ensemble members (each with n_mc dropout passes).
        # Ensemble epistemic uncertainty + MC dropout aleatoric uncertainty combined.
        all_pred_chunks: list[np.ndarray] = []
        active_nets = self.nets if self.nets else [self.net]
        for net_i in active_nets:
            X_rep = X_tensor.unsqueeze(0).expand(n_mc, -1, -1).reshape(n_mc * n_test, -1)
            net_i.train()  # keep dropout active for MC sampling
            with torch.no_grad():
                chunk = net_i(X_rep).reshape(n_mc, n_test).cpu().numpy()
            net_i.eval()
            all_pred_chunks.append(chunk)

        # shape: (n_ensemble * n_mc, n_test)
        predictions = np.concatenate(all_pred_chunks, axis=0)

        mean_mc = np.mean(predictions, axis=0)
        std_mc = np.std(predictions, axis=0, ddof=0)
        std_mc = np.where(np.isnan(std_mc), 1e-6, std_mc)

        # Apply temperature scaling in normalised space before unnormalising.
        # Corrects overconfidence so EI keeps pointing to unexplored regions.
        std_mc = std_mc * self._temp_scale

        if self.normalize_y:
            mean_mc = mean_mc * self._y_std + self._y_mean
            std_mc = std_mc * self._y_std

        std_mc = np.maximum(std_mc, 1e-6)

        return mean_mc.ravel(), std_mc.ravel()

    def run(self, objective, lb, ub, budget=1000, surrogate_interval=5, n_candidates=0,
            random_state=42, ec_type="lq", eval_fraction=0.1,
            eval_fraction_min=0.05, eval_fraction_max=0.5,
            lmm_n_iter_frac=0.05, lmm_tau_good=0.5,
            dts_beta=0.05, dts_epsilon_min=0.05, dts_epsilon_max=0.5,
            lq_tau_threshold=0.85,
            rmse_threshold=0.5, kappa_init=2.0, kappa_final=0.1) -> dict:
        """CMA-ES loop with BNN surrogate and evolution control from Pitra et al. (2021)."""
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
