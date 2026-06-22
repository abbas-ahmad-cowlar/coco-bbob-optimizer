"""
Transformer-based PFN with mixed prior for Bayesian Optimization.

Implements the core ideas from:
  PFNs4BO: In-Context Learning for Bayesian Optimization (Müller et al., 2023)

Architecture:
  - Context tokens (x_i, y_i) and query tokens (x_test) fed through a Transformer.
  - Causal masking: queries attend to context only, not to each other.
  - Output: Gaussian (mean, log_std) per query point.

Priors (mixed during pretraining):
  - BNN prior    (Section 4.3): random BNN architecture + weights → synthetic tasks.
  - GP/RBF prior (Section 4.1): y ~ N(0, K_RBF), random lengthscale.
  - Matérn 3/2   (Section 4.2): y ~ N(0, K_Matérn), random ϕ ~ p(ϕ;ψ).
  - Trained ONCE offline; weights saved to models/weights/pfn_bnn_transformer.pt.
  - Subsequent runs load saved weights — zero pretraining cost.

BO usage:
  fit(X, y)      → store context (no gradient updates)
  predict(X_q)   → single Transformer forward pass → (mean, std)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_WEIGHTS_PATH = Path(__file__).parent / "weights" / "pfn_bnn_transformer.pt"
_MAX_INPUT_DIM = 20   # pad all inputs to this size; handles dim=1..20
_N_BINS = 64          # reserved for future Riemann head; unused with Gaussian head


# ---------------------------------------------------------------------------
# BNN Prior: synthetic dataset sampler
# ---------------------------------------------------------------------------

def _sample_bnn_dataset(
    input_dim: int,
    n_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample a regression dataset from a random Bayesian Neural Network.

    Steps (per Section 4.3 of the paper):
      1. Sample random architecture: n_layers in {1,2,3}, hidden in {16,32,64}
      2. Sample weights from N(0, 1/sqrt(fan_in))
      3. Sample x uniformly from [0,1]^d
      4. Forward pass → y (with small Gaussian noise)
      5. Normalise y to zero mean / unit std
    """
    n_layers = int(rng.integers(1, 4))
    hidden_sizes = [int(rng.choice([16, 32, 64])) for _ in range(n_layers)]
    dims = [input_dim] + hidden_sizes + [1]

    # Sample weights
    layers: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(dims) - 1):
        fan_in = dims[i]
        W = rng.standard_normal((dims[i + 1], fan_in)) * (1.0 / max(np.sqrt(fan_in), 1e-8))
        b = rng.standard_normal(dims[i + 1]) * 0.1
        layers.append((W, b))

    # Sample inputs
    X = rng.uniform(0.0, 1.0, size=(n_points, input_dim))

    # Forward pass with tanh activations (more stable than ReLU for deep random nets)
    h = X.copy()
    for i, (W, b) in enumerate(layers):
        h = h @ W.T + b
        if i < len(layers) - 1:
            h = np.tanh(h)
    y = h.ravel()

    # Add small noise
    noise_std = float(rng.uniform(0.01, 0.15))
    y = y + rng.standard_normal(n_points) * noise_std

    # Normalise
    y_std = float(np.std(y))
    y = (y - float(np.mean(y))) / max(y_std, 1e-8)

    return X, y


# ---------------------------------------------------------------------------
# GP/RBF Prior: Section 4.1
# ---------------------------------------------------------------------------

def _sample_gp_dataset(
    input_dim: int,
    n_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample a regression dataset from a GP with RBF kernel (Section 4.1).

    Steps:
      1. Sample x ~ Uniform[0,1]^d
      2. Compute K_ij = exp(-||xi-xj||^2 / (2*l^2)),  l ~ Uniform[0.1, 1.0]
      3. Sample y ~ N(0, K)  via Cholesky
      4. Normalise y to zero mean / unit std
    """
    X = rng.uniform(0.0, 1.0, size=(n_points, input_dim))
    lengthscale = float(rng.uniform(0.1, 1.0))

    dists_sq = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    K = np.exp(-0.5 * dists_sq / (lengthscale ** 2)) + np.eye(n_points) * 1e-6

    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        K += np.eye(n_points) * 1e-4
        L = np.linalg.cholesky(K)

    y = L @ rng.standard_normal(n_points)
    y = (y - float(np.mean(y))) / max(float(np.std(y)), 1e-8)

    return X, y


# ---------------------------------------------------------------------------
# Matérn 3/2 Prior: Section 4.2
# ---------------------------------------------------------------------------

def _sample_matern_dataset(
    input_dim: int,
    n_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample a regression dataset from a GP with Matérn 3/2 kernel (Section 4.2).

    Hyperparameters ϕ ~ p(ϕ;ψ):
      - lengthscale ~ Uniform[0.1, 1.5]
      - outputscale ~ Uniform[0.5, 2.0]
      - noise       ~ Uniform[0.01, 0.1]

    K_ij = outputscale^2 * (1 + sqrt(3)*r/l) * exp(-sqrt(3)*r/l) + noise*I
    """
    X = rng.uniform(0.0, 1.0, size=(n_points, input_dim))
    lengthscale = float(rng.uniform(0.1, 1.5))
    outputscale = float(rng.uniform(0.5, 2.0))
    noise = float(rng.uniform(0.01, 0.1))

    dists = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1))
    r = np.sqrt(3.0) * dists / lengthscale
    K = outputscale ** 2 * (1.0 + r) * np.exp(-r) + np.eye(n_points) * (noise + 1e-6)

    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        K += np.eye(n_points) * 1e-4
        L = np.linalg.cholesky(K)

    y = L @ rng.standard_normal(n_points)
    y = (y - float(np.mean(y))) / max(float(np.std(y)), 1e-8)

    return X, y


# ---------------------------------------------------------------------------
# Transformer model
# ---------------------------------------------------------------------------

class PFNTransformer(nn.Module):
    """
    Transformer PFN for in-context regression.

    Sequence layout: [ctx_0 ... ctx_{n-1} | q_0 ... q_{m-1}]

    Attention masking:
      - Context → context:  allowed
      - Query   → context:  allowed  (queries condition on observations)
      - Query   → query:    blocked  (queries are independent given context)
      - Context → query:    blocked  (context does not peek at queries)
    """

    def __init__(
        self,
        max_input_dim: int = _MAX_INPUT_DIM,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_input_dim = max_input_dim
        self.d_model = d_model

        # Context encoder: (x padded, y) → d_model
        self.ctx_proj = nn.Linear(max_input_dim + 1, d_model)
        # Query encoder: x padded → d_model  (y unknown → not included)
        self.query_proj = nn.Linear(max_input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Gaussian output head: (mean, log_std) in normalised y-space
        self.output_head = nn.Linear(d_model, 2)

    def _build_mask(self, n_ctx: int, n_q: int, device: torch.device) -> torch.Tensor:
        n = n_ctx + n_q
        mask = torch.zeros(n, n, device=device)
        # Block context→query attention
        mask[:n_ctx, n_ctx:] = float("-inf")
        # Block query→query attention
        if n_q > 0:
            mask[n_ctx:, n_ctx:] = float("-inf")
        return mask

    def _pad(self, X: torch.Tensor, target_dim: int) -> torch.Tensor:
        d = X.shape[-1]
        if d < target_dim:
            pad = torch.zeros(*X.shape[:-1], target_dim - d, device=X.device, dtype=X.dtype)
            return torch.cat([X, pad], dim=-1)
        return X[..., :target_dim]

    def forward(
        self,
        X_ctx: torch.Tensor,    # (B, n_ctx, input_dim)  or  (n_ctx, input_dim)
        y_ctx: torch.Tensor,    # (B, n_ctx, 1)          or  (n_ctx, 1)
        X_query: torch.Tensor,  # (B, n_q, input_dim)    or  (n_q, input_dim)
    ) -> torch.Tensor:          # (B, n_q, 2)  — (mean, log_std)
        unbatched = X_ctx.dim() == 2
        if unbatched:
            X_ctx = X_ctx.unsqueeze(0)
            y_ctx = y_ctx.unsqueeze(0)
            X_query = X_query.unsqueeze(0)

        B, n_ctx = X_ctx.shape[:2]
        n_q = X_query.shape[1]
        device = X_ctx.device

        # Pad inputs to max_input_dim
        X_ctx_p = self._pad(X_ctx, self.max_input_dim)
        X_q_p = self._pad(X_query, self.max_input_dim)

        # Encode tokens
        ctx_tokens = self.ctx_proj(torch.cat([X_ctx_p, y_ctx], dim=-1))  # (B, n_ctx, d)
        q_tokens = self.query_proj(X_q_p)                                 # (B, n_q,   d)

        # Concatenate and apply Transformer with masking
        tokens = torch.cat([ctx_tokens, q_tokens], dim=1)                 # (B, n, d)
        mask = self._build_mask(n_ctx, n_q, device)                       # (n, n)
        out = self.transformer(tokens, mask=mask)                          # (B, n, d)

        # Extract and decode query positions
        query_repr = out[:, n_ctx:, :]                                     # (B, n_q, d)
        predictions = self.output_head(query_repr)                         # (B, n_q, 2)

        return predictions.squeeze(0) if unbatched else predictions


# ---------------------------------------------------------------------------
# Pretraining
# ---------------------------------------------------------------------------

# Prior type weights: (BNN=50%, GP/RBF=25%, Matérn=25%)
_PRIOR_TYPES = ["bnn", "gp", "matern"]
_PRIOR_WEIGHTS = [0.50, 0.25, 0.25]


def _sample_dataset(
    prior: str,
    input_dim: int,
    n_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to the appropriate prior sampler."""
    if prior == "bnn":
        return _sample_bnn_dataset(input_dim, n_points, rng)
    elif prior == "gp":
        return _sample_gp_dataset(input_dim, n_points, rng)
    else:  # matern
        return _sample_matern_dataset(input_dim, n_points, rng)


def pretrain_pfn(
    model: PFNTransformer,
    n_steps: int = 5000,
    episodes_per_step: int = 16,
    lr: float = 1e-3,
    device: torch.device | None = None,
    save_path: Path | None = None,
    verbose: bool = True,
) -> None:
    """
    Pretrain PFN on mixed prior datasets (BNN 50%, GP/RBF 25%, Matérn 3/2 25%).

    Per Section 4.1–4.3 of PFNs4BO (Müller et al., 2023):
    Each step samples `episodes_per_step` random functions (prior chosen randomly),
    splits each into context + query, and minimises Gaussian NLL on query predictions.
    """
    if device is None:
        device = torch.device("cpu")
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)
    rng = np.random.default_rng(0)

    if verbose:
        print(
            f"[PFN] Pretraining on mixed prior (BNN 50% + GP 25% + Matérn 25%): "
            f"{n_steps} steps × {episodes_per_step} episodes/step ...",
            flush=True,
        )

    for step in range(n_steps):
        total_loss = torch.tensor(0.0, device=device)
        count = 0

        optimizer.zero_grad()

        for _ in range(episodes_per_step):
            input_dim = int(rng.integers(1, min(6, model.max_input_dim + 1)))
            n_total = int(rng.integers(15, 80))
            prior = rng.choice(_PRIOR_TYPES, p=_PRIOR_WEIGHTS)
            X, y = _sample_dataset(prior, input_dim, n_total, rng)

            n_ctx = int(rng.integers(5, max(6, n_total - 3)))
            n_ctx = min(n_ctx, n_total - 3)
            idx = rng.permutation(n_total)
            ctx_idx = idx[:n_ctx]
            query_idx = idx[n_ctx:]
            if len(query_idx) == 0:
                continue

            X_ctx = torch.tensor(X[ctx_idx], dtype=torch.float32, device=device)
            y_ctx = torch.tensor(y[ctx_idx], dtype=torch.float32, device=device).unsqueeze(-1)
            X_q = torch.tensor(X[query_idx], dtype=torch.float32, device=device)
            y_q = torch.tensor(y[query_idx], dtype=torch.float32, device=device)

            pred = model(X_ctx, y_ctx, X_q)            # (n_q, 2)
            mean = pred[:, 0]
            log_std = pred[:, 1].clamp(-4.0, 4.0)
            std = torch.exp(log_std)

            # Gaussian NLL
            nll = 0.5 * ((y_q - mean) / std) ** 2 + log_std
            total_loss = total_loss + nll.mean()
            count += 1

        if count > 0:
            (total_loss / count).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        if verbose and (step + 1) % 500 == 0:
            loss_val = float(total_loss.detach() / max(count, 1))
            print(f"  step {step + 1:5d}/{n_steps} | loss={loss_val:.4f}", flush=True)

    model.eval()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        if verbose:
            print(f"[PFN] Weights saved → {save_path}", flush=True)


# ---------------------------------------------------------------------------
# Surrogate wrapper
# ---------------------------------------------------------------------------

class PFNTransformerSurrogate:
    """
    Transformer PFN surrogate for Bayesian Optimization.

    - Weights loaded from disk on init (or pretrained once and saved).
    - fit() stores normalised context — NO gradient updates.
    - predict() is a single Transformer forward pass.
    """

    in_context: bool = True

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        n_pretrain_steps: int = 5000,
        random_state: int | None = None,
        device: str | None = None,
    ) -> None:
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.n_pretrain_steps = n_pretrain_steps
        self.random_state = random_state
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self._model: PFNTransformer | None = None
        self._X_ctx: torch.Tensor | None = None
        self._y_ctx: torch.Tensor | None = None
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._fitted: bool = False

        self._load_or_pretrain()

    def _build_model(self) -> PFNTransformer:
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        return PFNTransformer(
            max_input_dim=_MAX_INPUT_DIM,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
        )

    def _load_or_pretrain(self) -> None:
        # Check for saved architecture config alongside weights
        config_path = _WEIGHTS_PATH.with_suffix(".yaml")
        if _WEIGHTS_PATH.exists() and config_path.exists():
            import yaml
            with config_path.open("r") as f:
                arch = yaml.safe_load(f)
            self.d_model         = arch.get("d_model",         self.d_model)
            self.nhead           = arch.get("nhead",           self.nhead)
            self.num_layers      = arch.get("num_layers",      self.num_layers)
            self.dim_feedforward = arch.get("dim_feedforward", self.dim_feedforward)

        self._model = self._build_model()
        if _WEIGHTS_PATH.exists():
            self._model.load_state_dict(
                torch.load(_WEIGHTS_PATH, map_location=self.device, weights_only=True)
            )
            self._model = self._model.to(self.device)
            self._model.eval()
        else:
            pretrain_pfn(
                self._model,
                n_steps=self.n_pretrain_steps,
                device=self.device,
                save_path=_WEIGHTS_PATH,
                verbose=True,
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Store normalised context — no gradient updates."""
        X = np.atleast_2d(X)
        y = np.asarray(y).ravel()

        # Normalise X
        self._x_mean = X.mean(axis=0)
        self._x_std = np.where(X.std(axis=0) < 1e-8, 1.0, X.std(axis=0))
        X_norm = (X - self._x_mean) / self._x_std

        # Normalise y (range-aware to survive near optimum)
        self._y_mean = float(np.mean(y))
        y_std_raw = float(np.std(y))
        y_range = float(np.max(y) - np.min(y))
        self._y_std = max(y_std_raw, 0.01 * y_range, 1e-8)
        y_norm = (y - self._y_mean) / self._y_std

        self._X_ctx = torch.tensor(X_norm, dtype=torch.float32, device=self.device)
        self._y_ctx = torch.tensor(y_norm, dtype=torch.float32, device=self.device).unsqueeze(-1)
        self._fitted = True

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Single Transformer forward pass — returns (mean, std) in original y-space."""
        if not self._fitted or self._model is None:
            raise RuntimeError("Call fit() before predict().")

        X = np.atleast_2d(X)
        X_norm = (X - self._x_mean) / self._x_std
        X_tensor = torch.tensor(X_norm, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            pred = self._model(self._X_ctx, self._y_ctx, X_tensor)  # (n_q, 2)

        mean_norm = pred[:, 0].cpu().numpy()
        log_std_norm = pred[:, 1].clamp(-4.0, 4.0).cpu().numpy()
        std_norm = np.exp(log_std_norm)

        # Unnormalise
        mean = mean_norm * self._y_std + self._y_mean
        std = np.maximum(std_norm * self._y_std, 1e-6)

        return mean.ravel(), std.ravel()


# ---------------------------------------------------------------------------
# Standalone pretraining entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pretrain PFN Transformer with BNN prior.")
    parser.add_argument("--steps", type=int, default=10000, help="Number of training steps.")
    parser.add_argument("--episodes", type=int, default=16, help="Episodes per step.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--device", type=str, default=None, help="Device: cpu, cuda, mps.")
    parser.add_argument("--force", action="store_true", help="Retrain even if weights exist.")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print loss every 500 steps.")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false", help="Suppress training output.")
    args = parser.parse_args()

    if _WEIGHTS_PATH.exists() and not args.force:
        print(f"[PFN] Weights already exist at {_WEIGHTS_PATH}. Use --force to retrain.")
    else:
        if args.device:
            device = torch.device(args.device)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        print(f"[PFN] Device: {device}")
        model = PFNTransformer()
        pretrain_pfn(
            model,
            n_steps=args.steps,
            episodes_per_step=args.episodes,
            lr=args.lr,
            device=device,
            save_path=_WEIGHTS_PATH,
            verbose=args.verbose,
        )
