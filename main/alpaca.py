import math
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, vmap, lax, jit, value_and_grad
from functools import partial

import flax.linen as nn
import optax


class MLP(nn.Module):
    in_dim: int
    layer_sizes: Tuple[int, ...]
    activation: str = 'relu'

    @nn.compact
    def __call__(self, x):
        act_map = {
            'relu': nn.relu,
            'tanh': nn.tanh,
            'sigmoid': nn.sigmoid,
        }
        act = act_map[self.activation]
        sizes = (self.in_dim,) + tuple(self.layer_sizes)
        out = x
        for i in range(len(sizes) - 1):
            out = nn.Dense(sizes[i + 1])(out)
            if i < len(sizes) - 1:
                out = act(out)
        return out


class ALPaCAjax:
    """
    JAX/Flax + Optax port of the ALPaCA module from the PyTorch implementation.

    This is structured as a stateful helper that holds optimizer state and params
    but exposes functional forward/loss/train_step interfaces.

    Usage:
      model = ALPaCAjax(config, preprocess=..., f_nom=..., rng=...)
      params = model.init_params()
      opt_state = model.init_optimizer(params)
      outputs = model.forward(params, context_x, context_y, x, y, num_context)
      params, opt_state, metrics = model.train_step(params, opt_state, batch, rng)
    """

    def __init__(self, config: Dict[str, Any], preprocess: Optional[Callable] = None,
                 f_nom: Optional[Callable] = None, rng: Optional[jax.random.PRNGKey] = None):
        self.config = deepcopy(config)
        self.lr = float(config['lr'])
        self.x_dim = int(config['x_dim'])
        self.y_dim = int(config['y_dim'])
        self.phi_dim = int(config['nn_layers'][-1])
        self.sigma_eps_cfg = config['sigma_eps']
        self.preprocess = preprocess
        self.f_nom = f_nom

        self.mlp = MLP(self.x_dim, tuple(config['nn_layers']), activation=config.get('activation', 'relu'))

        # SigEps (fixed)
        if isinstance(self.sigma_eps_cfg, (list, tuple, np.ndarray)):
            diag = jnp.asarray(self.sigma_eps_cfg, dtype=jnp.float32)
            SigEps = jnp.diag(diag)
        else:
            SigEps = jnp.eye(self.y_dim, dtype=jnp.float32) * float(self.sigma_eps_cfg)
        self.SigEps = SigEps
        self.SigEps_inv = jnp.linalg.inv(SigEps)
        sign, logabsdet = jnp.linalg.slogdet(SigEps)
        self.logdet_SigEps = (sign * logabsdet)  # sign should be +1 for PSD

        # RNG for param init
        self.rng = random.PRNGKey(0) if rng is None else rng

    def init_params(self, rng: Optional[jax.random.PRNGKey] = None):
        rng = self.rng if rng is None else rng
        rng_phi, rng_K, rng_L = random.split(rng, 3)

        # init phi_net params via flax init; needs a dummy input
        dummy_x = jnp.zeros((1, self.x_dim), dtype=jnp.float32)
        phi_params = self.mlp.init(rng_phi, dummy_x)

        # Initialize K (phi_dim, y_dim) with Xavier-like normal
        K = random.normal(rng_K, (self.phi_dim, self.y_dim), dtype=jnp.float32) * math.sqrt(2.0 / (self.phi_dim + self.y_dim))

        # Initialize L_asym (phi_dim, phi_dim) roughly like kaiming uniform
        bound = math.sqrt(6.0 / self.phi_dim)
        L_asym = random.uniform(rng_L, (self.phi_dim, self.phi_dim), minval=-bound, maxval=bound, dtype=jnp.float32)

        params = {
            'phi': phi_params,
            'K': K,
            'L_asym': L_asym,
        }
        return params

    def init_optimizer(self, params):
        opt = optax.adam(self.lr)
        opt_state = opt.init(params)
        self._opt = opt
        return opt_state

    def L(self, params):
        # PSD precision prior L = L_asym @ L_asym^T
        La = params['L_asym']
        return La @ La.T

    def basis_apply(self, params, x):
        # x: (..., x_dim)
        inp = self.preprocess(x) if self.preprocess is not None else x
        return self.mlp.apply(params['phi'], inp)

    def _blr_posterior_single(self, context_phi_b, context_y_b, n, params):
        """
        Compute posterior for single batch element (vmap target).
        context_phi_b: (Nc, F)
        context_y_b:   (Nc, y_dim)
        n: scalar int (# effective context rows)
        returns Kn: (F, y_dim), Ln_inv: (F, F)
        """
        F = self.phi_dim
        L = self.L(params)
        eyeF = jnp.eye(F, dtype=context_phi_b.dtype)
        jitter = 1e-6

        def with_data(args):
            Xb, Yb = args
            XtX = Xb.T @ Xb
            Ln = XtX + L + jitter * eyeF
            Ln_inv = jnp.linalg.inv(Ln)
            Kn = Ln_inv @ (Xb.T @ Yb + L @ params['K'])
            return Kn, Ln_inv

        def no_data(_):
            Ln_inv = jnp.linalg.inv(L + jitter * eyeF)
            Kn = params['K']
            return Kn, Ln_inv

        # Gather first n rows safely using take on dynamic range
        idx = jnp.arange(n, dtype=jnp.int32)
        Xb = lax.cond(n > 0, lambda i: context_phi_b.take(i, axis=0), lambda i: jnp.zeros((0, F), dtype=context_phi_b.dtype), idx)
        Yb = lax.cond(n > 0, lambda i: context_y_b.take(i, axis=0), lambda i: jnp.zeros((0, self.y_dim), dtype=context_y_b.dtype), idx)

        Kn, Ln_inv = lax.cond(n > 0, with_data, no_data, (Xb, Yb))
        return Kn, Ln_inv

    def _blr_posterior(self, context_phi, context_y, num_context, params):
        """
        Vectorized BLR posterior over batch.
        context_phi: (B, Nc, F)
        context_y:   (B, Nc, y_dim)
        num_context: (B,)
        returns K_post: (B, F, y_dim), L_inv_post: (B, F, F)
        """
        blr_single = lambda ph, yy, n: self._blr_posterior_single(ph, yy, n, params)
        K_post, L_inv_post = vmap(blr_single, in_axes=(0, 0, 0))(context_phi, context_y, num_context)
        return K_post, L_inv_post

    def _predictive(self, phi, K_post, L_inv_post, params, f_nom_x=None):
        """
        phi: (B, T, F)
        K_post: (B, F, y)
        L_inv_post: (B, F, F)
        f_nom_x: (B, T, y) or None
        """
        # mu = einsum('btf,bfy->bty')
        mu = jnp.einsum('btf,bfy->bty', phi, K_post)

        if f_nom_x is not None:
            mu = mu + f_nom_x

        # s = diag(phi L_inv phi^T) via einsum
        s = jnp.einsum('btf,bfg,btg->bt', phi, L_inv_post, phi)
        spread_fac = 1.0 + s  # (B, T)
        Sig_pred = spread_fac[..., None, None] * self.SigEps  # (B, T, y, y)
        return mu, Sig_pred, spread_fac

    def forward(self, params, context_x, context_y, x, y=None, num_context=None):
        """
        params: dict of parameters
        Inputs as jnp arrays:
          context_x: (B, Nc, x_dim)
          context_y: (B, Nc, y_dim)
          x: (B, T, x_dim)
          y: (B, T, y_dim) or None
          num_context: (B,) int32
        Returns:
          mu_pred (B,T,y), Sig_pred (B,T,y,y), predictive_nll (B,T) or None, aux dict
        """
        B = context_x.shape[0]

        # Encode
        context_phi = self.basis_apply(params, context_x)  # (B, Nc, F)
        phi = self.basis_apply(params, x)                  # (B, T, F)

        # f_nom
        f_nom_cx = jnp.zeros_like(context_y)
        f_nom_x = jnp.zeros((x.shape[0], x.shape[1], self.y_dim), dtype=x.dtype)
        if self.f_nom is not None:
            f_nom_cx = self.f_nom(context_x)
            f_nom_x = self.f_nom(x)

        context_y_blr = context_y - f_nom_cx

        # Ensure num_context has shape (B,)
        if num_context is None:
            num_context = jnp.full((B,), context_phi.shape[1], dtype=jnp.int32)
        else:
            num_context = num_context.astype(jnp.int32)

        K_post, L_inv_post = self._blr_posterior(context_phi, context_y_blr, num_context, params)
        mu_pred, Sig_pred, spread_fac = self._predictive(phi, K_post, L_inv_post, params, f_nom_x=f_nom_x)

        predictive_nll = None
        aux = {}
        if y is not None:
            log_spread = jnp.log(jnp.clip(spread_fac, a_min=1e-12))
            logdet = self.y_dim * log_spread + self.logdet_SigEps  # (B, T)

            alpha = jnp.clip(1.0 / spread_fac, a_max=1e12)
            resid = (y - mu_pred)  # (B, T, y)
            quad_core = jnp.einsum('bti,ij,btj->bt', resid, self.SigEps_inv, resid)  # (B, T)
            quadf = alpha * quad_core
            predictive_nll = logdet + quadf

            rmse_1 = jnp.sqrt(jnp.sum((mu_pred[:, 0, :] - y[:, 0, :]) ** 2, axis=-1)).mean()
            mpv_1 = jnp.mean((spread_fac[:, 0] ** self.y_dim) * jnp.exp(self.logdet_SigEps))
            aux = {'RMSE_1step': float(rmse_1), 'MPV_1step': float(mpv_1)}

        return mu_pred, Sig_pred, predictive_nll, aux

    def loss(self, predictive_nll):
        return jnp.mean(predictive_nll)

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, params, opt_state, batch, rng=None):
        """
        One training update step.

        batch is a dict with keys:
          'x': (B, N, x_dim), 'y': (B, N, y_dim)

        It follows the same split logic as the PyTorch version using
        config['data_horizon'] and config['test_horizon'].
        """
        batch_size = int(self.config['meta_batch_size'])
        horizon = int(self.config['data_horizon'])
        test_horizon = int(self.config['test_horizon'])

        x_np = batch['x']  # assumed to be jnp arrays already (B, N, x_dim)
        y_np = batch['y']

        # split
        ctx_x = x_np[:, :horizon, :]
        ctx_y = y_np[:, :horizon, :]
        tgt_x = x_np[:, horizon:horizon + test_horizon, :]
        tgt_y = y_np[:, horizon:horizon + test_horizon, :]

        # random num_context per batch element in [0, horizon]
        rng_local = self.rng if rng is None else rng
        rng_local, rsub = random.split(rng_local)
        num_context_np = random.randint(rsub, (batch_size,), 0, horizon + 1, dtype=jnp.int32)

        def loss_fn(params_local):
            _, _, predictive_nll, aux = self.forward(params_local, ctx_x, ctx_y, tgt_x, y=tgt_y, num_context=num_context_np)
            loss_val = self.loss(predictive_nll)
            return loss_val, aux

        (loss_val, aux), grads = value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = self._opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        metrics = {'loss': float(loss_val), **aux}
        return new_params, new_opt_state, metrics

    # convenience
    def encode(self, params, x):
        return self.basis_apply(params, x)

    def test(self, params, x_c, y_c, x, num_context=None):
        mu_pred, Sig_pred, _, _ = self.forward(params, x_c, y_c, x, y=None, num_context=num_context)