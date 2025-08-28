from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from flax import struct
import optax
import numpy as np


# -------------------- MLP --------------------

def _get_activation(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    name = name.lower()
    if name == 'relu':
        return nn.relu
    if name == 'tanh':
        return jnp.tanh
    if name == 'sigmoid':
        return jax.nn.sigmoid
    raise ValueError(f'Unsupported activation: {name}')


class MLP(nn.Module):
    in_dim: int
    layer_sizes: Sequence[int]
    activation: str = 'relu'

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        act = _get_activation(self.activation)
        out = x
        for i, size in enumerate(self.layer_sizes):
            out = nn.Dense(size, name=f'dense_{i}')(out)
            if i < len(self.layer_sizes) - 1:
                out = act(out)
        return out


# -------------------- ALPaCA --------------------

class ALPaCA(nn.Module):
    """
    JAX/Flax rewrite of ALPaCA with BLR last layer.

    Expected shapes:
      - context_x: (B, N_context, x_dim)
      - context_y: (B, N_context, y_dim)
      - x: (B, T, x_dim)
      - y: (B, T, y_dim) [optional for inference]
      - num_context: (B,) int array with number of effective context points per batch entry

    Predictive distribution:
      mu_pred = phi(x) @ K_post  + f_nom(x)
      Sig_pred = (1 + phi L_inv_post phi^T) * Sigma_eps
    """
    config: Dict[str, Any]
    preprocess: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
    f_nom: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None

    lr: float = struct.field(pytree_node=False)
    x_dim: int = struct.field(pytree_node=False)
    y_dim: int = struct.field(pytree_node=False)
    nn_layers: Sequence[int] = struct.field(pytree_node=False)
    phi_dim: int = struct.field(pytree_node=False)
    activation: str = struct.field(pytree_node=False)
    sigma_eps_cfg: Any = struct.field(pytree_node=False)  # float or list/array of floats

    def setup(self):
        self.lr = float(self.config['lr'])
        self.x_dim = int(self.config['x_dim'])
        self.y_dim = int(self.config['y_dim'])
        self.nn_layers = list(self.config['nn_layers'])
        self.phi_dim = int(self.nn_layers[-1])
        self.activation = self.config.get('activation', 'relu')
        self.sigma_eps_cfg = self.config['sigma_eps']

        # Basis network phi
        self.phi_net = MLP(self.x_dim, self.nn_layers, activation=self.activation)

        # BLR prior parameters for last layer: K (phi_dim, y_dim), L_asym (phi_dim, phi_dim)
        glorot_normal = nn.initializers.variance_scaling(1.0, 'fan_avg', 'truncated_normal')
        kaiming_uniform = nn.initializers.variance_scaling(2.0, 'fan_in', 'uniform')

        self.K = self.param('K', glorot_normal, (self.phi_dim, self.y_dim))
        self.L_asym = self.param('L_asym', kaiming_uniform, (self.phi_dim, self.phi_dim))

    def _SigEps(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # Observation noise covariance Sigma_eps (fixed)
        if isinstance(self.sigma_eps_cfg, (list, tuple, np.ndarray)):
            diag = jnp.asarray(self.sigma_eps_cfg, dtype=jnp.float32)
            SigEps = jnp.diag(diag)
        else:
            SigEps = jnp.eye(self.y_dim, dtype=jnp.float32) * float(self.sigma_eps_cfg)
        SigEps_inv = jnp.linalg.inv(SigEps)
        sign, logdet = jnp.linalg.slogdet(SigEps)
        # Sigma_eps must be PD => sign should be 1
        logdet = jnp.clip(logdet, a_min=-1e12, a_max=1e12)
        return SigEps, SigEps_inv, logdet

    def _L(self) -> jnp.ndarray:
        # Ensure PSD precision prior
        return self.L_asym @ self.L_asym.T

    def basis(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (..., x_dim) -> (..., phi_dim)
        inp = self.preprocess(x) if self.preprocess is not None else x
        return self.phi_net(inp)

    def _blr_posterior(
        self,
        context_phi: jnp.ndarray,    # (B, Nc, F)
        context_y: jnp.ndarray,      # (B, Nc, y)
        num_context: Optional[jnp.ndarray],  # (B,)
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute BLR posterior per batch with masked variable num_context.

        Returns:
          K_post:     (B, F, y)
          L_inv_post: (B, F, F)
        """
        B, Nc, F = context_phi.shape
        L = self._L()                              # (F,F)
        eyeF = jnp.eye(F, dtype=context_phi.dtype)
        jitter = 1e-6

        if num_context is None:
            mask = jnp.ones((B, Nc), dtype=context_phi.dtype)
        else:
            # mask[b, i] = 1 if i < num_context[b] else 0
            idx = jnp.arange(Nc)[None, :]
            mask = (idx < num_context[:, None]).astype(context_phi.dtype)  # (B,Nc)

        Xb = context_phi * mask[..., None]  # (B,Nc,F)
        Yb = context_y * mask[..., None]    # (B,Nc,y)

        XtX = jnp.einsum('bnf,bng->bfg', Xb, Xb)                     # (B,F,F)
        Ln = XtX + L + jitter * eyeF                                 # (B,F,F) + (F,F) -> broadcast
        Ln_inv = jnp.linalg.inv(Ln)                                  # (B,F,F)

        rhs = jnp.einsum('bnf,bny->bfy', Xb, Yb) + (L @ self.K)      # (B,F,y)
        K_post = jnp.einsum('bfg,bgy->bfy', Ln_inv, rhs)             # (B,F,y)
        return K_post, Ln_inv

    def _predictive(
        self,
        phi: jnp.ndarray,             # (B, T, F)
        K_post: jnp.ndarray,          # (B, F, y)
        L_inv_post: jnp.ndarray,      # (B, F, F)
        f_nom_x: Optional[jnp.ndarray] = None,  # (B, T, y)
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Returns:
          mu_pred:    (B, T, y)
          Sig_pred:   (B, T, y, y)
          spread_fac: (B, T)
        """
        mu = jnp.einsum('btf,bfy->bty', phi, K_post)  # (B,T,y)
        if f_nom_x is not None:
            mu = mu + f_nom_x

        s = jnp.einsum('btf,bfg,btg->bt', phi, L_inv_post, phi)  # (B,T)
        spread_fac = 1.0 + s

        SigEps, _, _ = self._SigEps()
        Sig_pred = spread_fac[..., None, None] * SigEps  # (B,T,y,y)
        return mu, Sig_pred, spread_fac

    def __call__(
        self,
        context_x: jnp.ndarray,     # (B, Nc, x)
        context_y: jnp.ndarray,     # (B, Nc, y)
        x: jnp.ndarray,             # (B, T, x)
        y: Optional[jnp.ndarray] = None,    # (B, T, y)
        num_context: Optional[jnp.ndarray] = None,  # (B,)
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray], Dict[str, float]]:
        """
        Runs full ALPaCA pipeline and optionally returns NLL if y is provided.

        Returns:
          mu_pred:        (B, T, y)
          Sig_pred:       (B, T, y, y)
          predictive_nll: (B, T) if y provided, else None
          aux: dict with RMSE_1step, MPV_1step if y provided
        """
        context_phi = self.basis(context_x)  # (B,Nc,F)
        phi = self.basis(x)                  # (B,T,F)

        # f_nom
        f_nom_cx = context_y * 0.0
        f_nom_x = x[..., :self.y_dim] * 0.0  # shape: (B,T,y)
        if self.f_nom is not None:
            f_nom_cx = self.f_nom(context_x)
            f_nom_x = self.f_nom(x)

        # BLR uses residuals (subtract f_nom on context)
        context_y_blr = context_y - f_nom_cx

        # Posterior
        K_post, L_inv_post = self._blr_posterior(context_phi, context_y_blr, num_context)

        # Predictive
        mu_pred, Sig_pred, spread_fac = self._predictive(phi, K_post, L_inv_post, f_nom_x=f_nom_x)

        predictive_nll = None
        aux: Dict[str, float] = {}
        if y is not None:
            _, SigEps_inv, logdet_SigEps = self._SigEps()

            logdet = self.y_dim * jnp.log(jnp.clip(spread_fac, a_min=1e-12)) + logdet_SigEps  # (B,T)

            alpha = 1.0 / jnp.clip(spread_fac, a_min=1e-12)  # (B,T)
            resid = (y - mu_pred)  # (B,T,y)
            quad_core = jnp.einsum('bti,ij,btj->bt', resid, SigEps_inv, resid)  # (B,T)
            quadf = alpha * quad_core

            predictive_nll = logdet + quadf  # (B,T)

            rmse_1 = jnp.sqrt(jnp.sum((mu_pred[:, 0, :] - y[:, 0, :]) ** 2, axis=-1)).mean()
            mpv_1 = jnp.mean((spread_fac[:, 0] ** self.y_dim) * jnp.exp(logdet_SigEps))
            aux = {'RMSE_1step': float(rmse_1), 'MPV_1step': float(mpv_1)}

        return mu_pred, Sig_pred, predictive_nll, aux

    # Convenience wrappers

    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.basis(x)

    def test(
        self,
        x_c: jnp.ndarray, y_c: jnp.ndarray,
        x: jnp.ndarray,
        num_context: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        mu_pred, Sig_pred, _, _ = self(context_x=x_c, context_y=y_c, x=x, y=None, num_context=num_context)
        return mu_pred, Sig_pred


# -------------------- Training helpers (JAX/Flax/Optax) --------------------

@struct.dataclass
class ALPaCAState(train_state.TrainState):
    pass


def create_train_state(
    rng: jax.random.PRNGKey,
    model: ALPaCA,
    sample_batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> ALPaCAState:
    context_x, context_y, x, y, num_context = sample_batch
    variables = model.init(rng, context_x, context_y, x, y, num_context)
    params = variables['params']
    tx = optax.adam(model.lr)
    return ALPaCAState.create(apply_fn=model.apply, params=params, tx=tx)


def _batch_from_numpy(
    x_np: np.ndarray, y_np: np.ndarray, horizon: int, test_horizon: int, rng: np.random.Generator
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    ctx_x_np = x_np[:, :horizon, :]
    ctx_y_np = y_np[:, :horizon, :]
    tgt_x_np = x_np[:, horizon:, :]
    tgt_y_np = y_np[:, horizon:, :]
    num_context_np = rng.integers(low=0, high=horizon + 1, size=x_np.shape[0], dtype=np.int32)

    context_x = jnp.asarray(ctx_x_np, dtype=jnp.float32)
    context_y = jnp.asarray(ctx_y_np, dtype=jnp.float32)
    x = jnp.asarray(tgt_x_np, dtype=jnp.float32)
    y = jnp.asarray(tgt_y_np, dtype=jnp.float32)
    num_context = jnp.asarray(num_context_np, dtype=jnp.int32)
    return context_x, context_y, x, y, num_context


def train_loop(
    model: ALPaCA,
    dataset,                      # expects: dataset.sample(n_funcs, n_samples) -> (x_np, y_np)
    num_train_updates: int,
    seed: int = 0,
) -> Tuple[ALPaCAState, Dict[str, Any]]:
    """
    dataset.sample(n_funcs, n_samples) should return numpy arrays:
      x: (B, N, x_dim), y: (B, N, y_dim)
    """
    batch_size = int(model.config['meta_batch_size'])
    horizon = int(model.config['data_horizon'])
    test_horizon = int(model.config['test_horizon'])

    # Bootstrap an example batch to init
    x_np, y_np = dataset.sample(n_funcs=batch_size, n_samples=horizon + test_horizon)
    np_rng = np.random.default_rng(seed=seed)
    init_batch = _batch_from_numpy(x_np, y_np, horizon, test_horizon, np_rng)

    rng = jax.random.PRNGKey(seed)
    state = create_train_state(rng, model, init_batch)

    @jax.jit
    def update_step(state: ALPaCAState, batch) -> Tuple[ALPaCAState, jnp.ndarray, Dict[str, float]]:
        context_x, context_y, x, y, num_context = batch

        def loss_fn(params):
            mu_pred, Sig_pred, predictive_nll, aux = model.apply(
                {'params': params}, context_x, context_y, x, y=y, num_context=num_context
            )
            loss = jnp.mean(predictive_nll)  # mean over batch and time
            return loss, aux

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss, aux

    loss_history: list[float] = []
    for i in range(num_train_updates):
        x_np, y_np = dataset.sample(n_funcs=batch_size, n_samples=horizon + test_horizon)
        batch = _batch_from_numpy(x_np, y_np, horizon, test_horizon, np_rng)

        state, loss, aux = update_step(state, batch)
        loss_val = float(loss)
        loss_history.append(loss_val)

        if i % 50 == 0:
            rmse = aux.get('RMSE_1step', None)
            mpv = aux.get('MPV_1step', None)
            if rmse is not None and mpv is not None:
                print(f'iter {i:6d}  loss: {loss_val:.6f}  RMSE_1step: {rmse:.6f}  MPV_1step: {mpv:.6f}')
            else:
                print(f'iter {i:6d}  loss: {loss_val:.6f}')

    return state, {'loss_history': loss_history}


# ---------- JAX equivalents of the helper functions ----------

def batch_matmul(mat: jnp.ndarray, batch_v: jnp.ndarray) -> jnp.ndarray:
    """
    mat: (..., N, N2)
    batch_v: (..., M, N2)
    returns: (..., M, N)
    """
    return jnp.matmul(mat, jnp.swapaxes(batch_v, -1, -2)).swapaxes(-1, -2)


def batch_quadform(A: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    A:  either (..., n, n) or (..., N, n, n)
    b:  (..., N, n)
    returns: (..., N, 1) where each entry is b^T A b
    """
    v = b[..., None]  # (..., N, n, 1)
    if A.ndim == b.ndim - 1:
        return jnp.matmul(jnp.swapaxes(v, -1, -2), jnp.matmul(A, v))
    elif A.ndim == b.ndim:
        return jnp.matmul(jnp.swapaxes(v, -1, -2), jnp.matmul(A, v))
    else:
        raise ValueError(f'Matrix size of {A.ndim} is not supported.')


def blr_update_jnp(K: jnp.ndarray, L: jnp.ndarray, X: jnp.ndarray, Y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Single (non-batched) BLR update in JAX.
    """
    Ln_inv = jnp.linalg.inv(X.T @ X + L)
    Kn = Ln_inv @ (X.T @ Y + L @ K)
    return Kn, Ln_inv


def sampleMN_jax(key: jax.random.PRNGKey, K: jnp.ndarray, L_inv: jnp.ndarray, Sig: jnp.ndarray) -> jnp.ndarray:
    """
    Sample K ~ MN(K, L_inv, Sig) using vec(K) ~ N(vec(K), Sig ⊗ L_inv)
    """
    mean = jnp.reshape(K.T, [-1])
    cov = jnp.kron(Sig, L_inv)
    sample = jax.random.multivariate_normal(key, mean, cov)
    return jnp.reshape(sample, K.T.shape).T


def tp(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.swapaxes(x, -1, -2)


def extract_x(xu: jnp.ndarray, x_dim: int) -> jnp.ndarray:
    """
    JAX version: slice the last dimension to first x_dim entries.
    xu: (..., D)
    returns: (..., x_dim)
    """
    return xu[..., :x_dim]


class TrainedALPaCA:
    """Lightweight wrapper that binds params to the model for easy .test() calls."""
    def __init__(self, model: ALPaCA, state: ALPaCAState, config):
        self.model = model
        self.state = state

    def test(self, x_c, y_c, x, num_context=None):
        mu_pred, Sig_pred, _, _ = self.model.apply(
            {'params': self.state.params},
            x_c, y_c, x, y=None, num_context=num_context
        )
        return mu_pred, Sig_pred

    def eval(self):
        # no-op (for compatibility with old code paths)
        return self