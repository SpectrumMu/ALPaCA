import math
from copy import deepcopy
import time
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, in_dim, layer_sizes, activation='relu'):
        super().__init__()
        acts = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
        }
        self.act = acts[activation]
        sizes = [in_dim] + list(layer_sizes)
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # x: (..., in_dim)
        out = x
        for i, lin in enumerate(self.layers):
            out = lin(out)
            out = self.act(out) if i < len(self.layers) - 1 else out
        return out


class ALPaCA(nn.Module):
    """
    PyTorch rewrite of ALPaCA with BLR last layer.

    Expected shapes:
      - context_x: (B, N_context, x_dim)
      - context_y: (B, N_context, y_dim)
      - x: (B, T, x_dim)
      - y: (B, T, y_dim) [target for loss; optional for inference]
      - num_context: (B,) int tensor with number of effective context points per batch entry

    Predictive distribution:
      mu_pred = phi(x) @ K_post  + f_nom(x)
      Sig_pred = (1 + phi L_inv_post phi^T) * Sigma_eps
    """
    def __init__(self, config, preprocess=None, f_nom=None, device=None):
        super().__init__()
        self.config = deepcopy(config)
        self.lr = config['lr']
        self.x_dim = config['x_dim']
        self.y_dim = config['y_dim']
        self.phi_dim = config['nn_layers'][-1]
        self.sigma_eps_cfg = config['sigma_eps']
        self.device = torch.device(device) if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Basis network phi
        self.preprocess = preprocess  # callable or None; expects torch tensor
        self.f_nom = f_nom            # callable or None; expects torch tensor
        self.phi_net = MLP(self.x_dim, config['nn_layers'], activation=config['activation'])

        # BLR prior parameters for last layer
        # K: (phi_dim, y_dim). L = L_asym L_asym^T: (phi_dim, phi_dim)
        self.K = nn.Parameter(torch.zeros(self.phi_dim, self.y_dim))
        nn.init.xavier_normal_(self.K)

        self.L_asym = nn.Parameter(torch.empty(self.phi_dim, self.phi_dim))
        nn.init.kaiming_uniform_(self.L_asym, a=math.sqrt(5))

        # Observation noise covariance Sigma_eps (fixed, non-learnable by default)
        if isinstance(self.sigma_eps_cfg, (list, tuple, np.ndarray)):
            diag = torch.as_tensor(self.sigma_eps_cfg, dtype=torch.float32)
            SigEps = torch.diag(diag)
        else:
            SigEps = torch.eye(self.y_dim, dtype=torch.float32) * float(self.sigma_eps_cfg)

        self.register_buffer('SigEps', SigEps)             # (y_dim, y_dim)
        self.register_buffer('SigEps_inv', torch.linalg.inv(SigEps))  # (y_dim, y_dim)
        self.register_buffer('logdet_SigEps', torch.logdet(SigEps).clamp(min=-1e12, max=1e12))

        self.to(self.device)

    def L(self):
        # Ensure PSD precision prior
        return self.L_asym @ self.L_asym.T

    def basis(self, x):
        # x: (..., x_dim) -> (..., phi_dim)
        inp = self.preprocess(x) if self.preprocess is not None else x
        return self.phi_net(inp)

    @staticmethod
    def _batch_select_prefix(X, n):
        # X: (Nc, ...) -> X[:n, ...]
        return X[:n]

    def _blr_posterior(self, context_phi, context_y, num_context=None):
        """
        Compute BLR posterior per batch element with optional variable num_context.

        Inputs:
          context_phi: (B, Nc, phi_dim)
          context_y:   (B, Nc, y_dim)
          num_context: (B,) or None

        Returns:
          K_post:     (B, phi_dim, y_dim)
          L_inv_post: (B, phi_dim, phi_dim)
        """
        B, Nc, F = context_phi.shape
        assert F == self.phi_dim

        L = self.L()
        eyeF = torch.eye(self.phi_dim, device=context_phi.device, dtype=context_phi.dtype)
        # Small jitter improves numerical stability if needed
        jitter = 1e-6

        K_posts = []
        L_inv_posts = []

        for b in range(B):
            n = int(num_context[b].item()) if num_context is not None else Nc
            Xb = context_phi[b, :n, :]          # (n, F)
            Yb = context_y[b, :n, :]            # (n, y_dim)

            if n > 0:
                XtX = Xb.T @ Xb                 # (F, F)
                Ln = XtX + L + jitter * eyeF
                Ln_inv = torch.linalg.inv(Ln)   # (F, F)
                Kn = Ln_inv @ (Xb.T @ Yb + L @ self.K)  # (F, y_dim)
            else:
                Ln_inv = torch.linalg.inv(L + jitter * eyeF)
                Kn = self.K

            K_posts.append(Kn)
            L_inv_posts.append(Ln_inv)

        K_post = torch.stack(K_posts, dim=0)           # (B, F, y_dim)
        L_inv_post = torch.stack(L_inv_posts, dim=0)   # (B, F, F)
        return K_post, L_inv_post

    def _predictive(self, phi, K_post, L_inv_post, f_nom_x=None):
        """
        Inputs:
          phi:        (B, T, phi_dim)
          K_post:     (B, phi_dim, y_dim)
          L_inv_post: (B, phi_dim, phi_dim)
          f_nom_x:    (B, T, y_dim) or None

        Returns:
          mu_pred:    (B, T, y_dim)
          Sig_pred:   (B, T, y_dim, y_dim)
          spread_fac: (B, T)  where Sig_pred = (1 + spread_fac) * SigEps
        """
        # mu = phi @ K_post
        # Use (B, y, f) x (B, f, T) -> (B, y, T) -> (B, T, y)
        mu = torch.matmul(K_post.transpose(1, 2), phi.transpose(1, 2)).transpose(1, 2)

        if f_nom_x is not None:
            mu = mu + f_nom_x

        # s = diag(phi L_inv phi^T) as scalar per (B,T)
        # einsum: (B,T,F),(B,F,F),(B,T,F) -> (B,T)
        s = torch.einsum('btf,bfg,btg->bt', phi, L_inv_post, phi)
        spread_fac = 1.0 + s  # (B, T)

        # Sig_pred = spread_fac[...,None,None] * SigEps
        Sig_pred = spread_fac.unsqueeze(-1).unsqueeze(-1) * self.SigEps  # (B, T, y, y)

        return mu, Sig_pred, spread_fac

    def forward(self, context_x, context_y, x, y=None, num_context=None):
        """
        Runs full ALPaCA pipeline and optionally returns NLL if y is provided.

        Returns:
          mu_pred:        (B, T, y_dim)
          Sig_pred:       (B, T, y_dim, y_dim)
          predictive_nll: (B, T) if y provided, else None
          aux: dict with rmse_1, mpv_1 if y provided
        """
        B = context_x.shape[0]
        device = self.device

        context_x = context_x.to(device)
        context_y = context_y.to(device)
        x = x.to(device)
        y = y.to(device) if y is not None else None
        if num_context is not None:
            num_context = num_context.to(device)

        # Encode to feature space
        context_phi = self.basis(context_x)  # (B, Nc, F)
        phi = self.basis(x)                  # (B, T, F)

        # f_nom
        f_nom_cx = torch.zeros_like(context_y)
        f_nom_x = torch.zeros((x.shape[0], x.shape[1], self.y_dim), device=device, dtype=x.dtype)
        if self.f_nom is not None:
            f_nom_cx = self.f_nom(context_x)
            f_nom_x = self.f_nom(x)

        # BLR uses residuals (subtract f_nom on context)
        context_y_blr = context_y - f_nom_cx

        # Posterior over last layer weights
        K_post, L_inv_post = self._blr_posterior(context_phi, context_y_blr, num_context=num_context)

        # Predictive distribution
        mu_pred, Sig_pred, spread_fac = self._predictive(phi, K_post, L_inv_post, f_nom_x=f_nom_x)

        predictive_nll = None
        aux = {}
        if y is not None:
            # logdet(Sig_pred) = y_dim * log(spread_fac) + logdet(SigEps)
            logdet = self.y_dim * torch.log(spread_fac.clamp_min(1e-12)) + self.logdet_SigEps  # (B, T)

            # Quad form: (y - mu)^T Sig_pred^{-1} (y - mu)
            # Sig_pred^{-1} = (1/spread_fac) * SigEps^{-1}
            alpha = (1.0 / spread_fac).clamp_max(1e12)  # (B, T)
            resid = (y - mu_pred)  # (B, T, y)
            quad_core = torch.einsum('bti,ij,btj->bt', resid, self.SigEps_inv, resid)  # (B, T)
            quadf = alpha * quad_core  # (B, T)

            predictive_nll = logdet + quadf  # (B, T)

            # Aux metrics (match TF summaries)
            rmse_1 = torch.sqrt(torch.sum((mu_pred[:, 0, :] - y[:, 0, :]) ** 2, dim=-1)).mean()
            # MPV_1step = mean(det(Sig_pred[:,0])) = mean((spread_fac[:,0]**y_dim) * det(SigEps))
            mpv_1 = torch.mean((spread_fac[:, 0] ** self.y_dim) * torch.exp(self.logdet_SigEps))
            aux = {'RMSE_1step': rmse_1.item(), 'MPV_1step': mpv_1.item()}

        return mu_pred, Sig_pred, predictive_nll, aux

    def loss(self, predictive_nll):
        # total loss = mean over batch and time
        return predictive_nll.mean()

    def configure_optim(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    # Convenience wrappers similar to the original API
    @torch.no_grad()
    def encode(self, x):
        x = x.to(self.device)
        return self.basis(x)

    @torch.no_grad()
    def test(self, x_c, y_c, x, num_context=None):
        mu_pred, Sig_pred, _, _ = self.forward(x_c, y_c, x, y=None, num_context=num_context)
        return mu_pred, Sig_pred

    # --------------- Training helpers ---------------

    def train_step(self, dataset, num_train_updates):
        """
        Train the ALPaCA model using samples from `dataset`.

        dataset.sample(n_funcs, n_samples) should return:
          x: (B, N, x_dim), y: (B, N, y_dim) as numpy arrays

        This function overrides nn.Module.train; it will still set the module
        to training mode by calling the base implementation.
        """
        # Ensure module is in training mode
        # super(ALPaCA, self).train(True)
        self.train()

        batch_size = self.config['meta_batch_size']
        horizon = self.config['data_horizon']
        test_horizon = self.config['test_horizon']

        # Lazy-create optimizer and step counter
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            self.optimizer = self.configure_optim()
        if not hasattr(self, 'updates_so_far'):
            self.updates_so_far = 0

        device = self.device
        loss_history = []

        for i in range(num_train_updates):
            # Sample a meta-batch of tasks/sequences
            x_np, y_np = dataset.sample(n_funcs=batch_size, n_samples=horizon + test_horizon)

            # Split into context and target segments
            ctx_x_np = x_np[:, :horizon, :]
            ctx_y_np = y_np[:, :horizon, :]
            tgt_x_np = x_np[:, horizon:, :]
            tgt_y_np = y_np[:, horizon:, :]

            # Random number of effective context points per batch element: [0, horizon]
            num_context_np = np.random.randint(horizon + 1, size=batch_size)

            # To torch (on the right device/dtype)
            context_x = torch.as_tensor(ctx_x_np, dtype=torch.float32, device=device)
            context_y = torch.as_tensor(ctx_y_np, dtype=torch.float32, device=device)
            x = torch.as_tensor(tgt_x_np, dtype=torch.float32, device=device)
            y = torch.as_tensor(tgt_y_np, dtype=torch.float32, device=device)
            num_context = torch.as_tensor(num_context_np, dtype=torch.long, device=device)

            # Forward + loss
            mu_pred, Sig_pred, predictive_nll, aux = self.forward(
                context_x, context_y, x, y=y, num_context=num_context
            )
            loss = self.loss(predictive_nll)

            # Backprop
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            # Logging
            loss_val = float(loss.detach().cpu().item())
            loss_history.append(loss_val)

            if i % 50 == 0:
                rmse = aux.get('RMSE_1step', None)
                mpv = aux.get('MPV_1step', None)
                if rmse is not None and mpv is not None:
                    print(f'iter {i:6d}  loss: {loss_val:.6f}  RMSE_1step: {rmse:.6f}  MPV_1step: {mpv:.6f}')
                else:
                    print(f'iter {i:6d}  loss: {loss_val:.6f}')

            self.updates_so_far += 1

        return {'loss_history': loss_history}




# ---------- Torch equivalents of the helper functions ----------

def batch_matmul(mat, batch_v):
    """
    mat: (..., N, N2)
    batch_v: (..., M, N2)
    returns: (..., M, N)
    """
    return torch.matmul(mat, batch_v.transpose(-1, -2)).transpose(-1, -2)


def batch_quadform(A, b):
    """
    A:  either (..., n, n) or (..., N, n, n)
    b:  (..., N, n)
    returns: (..., N, 1) where each entry is b^T A b
    """
    if A.ndim == b.ndim - 1:
        # A: (..., n, n), b: (..., N, n)
        v = b.unsqueeze(-1)  # (..., N, n, 1)
        return (v.transpose(-1, -2) @ A @ v)
    elif A.ndim == b.ndim:
        # A: (..., N, n, n), b: (..., N, n)
        v = b.unsqueeze(-1)  # (..., N, n, 1)
        return (v.transpose(-1, -2) @ A @ v)
    else:
        raise ValueError(f'Matrix size of {A.ndim} is not supported.')


def blr_update_np(K, L, X, Y):
    Ln_inv = np.linalg.inv(X.T @ X + L)
    Kn = Ln_inv @ (X.T @ Y + L @ K)
    return Kn, Ln_inv


def sampleMN(K, L_inv, Sig):
    mean = np.reshape(K.T, [-1])
    cov = np.kron(Sig, L_inv)
    K_vec = np.random.multivariate_normal(mean, cov)
    return np.reshape(K_vec, K.T.shape).T


def tp(x):
    return np.swapaxes(x, -1, -2)


def extract_x(xu, x_dim):
    """
    Torch version: slice the last dimension to first x_dim entries.
    xu: (..., D)
    returns: (..., x_dim)
    """
    return xu[..., :x_dim]