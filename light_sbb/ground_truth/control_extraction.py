"""Recover the SBB controls (drift and volatility) from a trained LightSBB model.

The model never parameterizes sigma: it learns an SB on the Y-process plus the map
X = Y + (1/beta) s(t, Y). Ito on that map gives both controls, evaluated at Y_t and
indexed on X_t, so the map is never inverted.
"""

import copy

import torch

SAFE_T = 1e-2


def score(model, t, y):
    """Evaluate the Y-process SB drift, differentiably in `t` and `y`.

    `LightSBM.get_drift` detaches its input, so it cannot be differentiated with
    respect to the state; this mirrors it while keeping the graph intact.

    Args:
        model: trained LightSBM, diagonal-covariance branch only.
        t: (n,) times in [0, 1).
        y: (n, d) states.

    Returns:
        (n, d) drift, matching `model.get_drift(t, y)` numerically.
    """
    assert model.is_diagonal, "score() only implements the diagonal-covariance branch"
    eps = model.epsilon
    r = model.get_r()
    s_diag = torch.exp(model.S_log_diagonal_matrix)

    a = (t / (eps * (1 - t)))[:, None, None] + 1 / (eps * s_diag)[None, :, :]
    c = ((1 / (eps * (1 - t)))[:, None] * y)[:, None, :] + (r / (eps * s_diag))[None, :, :]

    exp_arg = (
        model.log_alpha[None, :]
        - 0.5 * torch.sum(model.S_log_diagonal_matrix, dim=-1)[None, :]
        - 0.5 * torch.sum(torch.log(a), dim=-1)
        - 0.5 * ((r * (1 / s_diag) * r) / eps).sum(dim=-1)[None, :]
        + 0.5 * (c * (1 / a) * c).sum(dim=-1)
    )
    lse = torch.logsumexp(exp_arg, dim=-1)
    grad = torch.autograd.grad(lse, y, grad_outputs=torch.ones_like(lse), create_graph=True)[0]
    return -y / (1 - t[:, None]) + eps * grad


def clamp_t(t, safe_t=SAFE_T):
    """Keep t away from 1, where the 1/(1-t) drift term blows up."""
    return min(float(t), 1.0 - safe_t)


def _as_double(model, t, y):
    """Promote model and inputs to float64 for the derivative computations.

    The Laplacian carries a 1/(1-t)^2 factor, so float32 loses roughly two digits
    per halving of (1-t) and reaches percent-level error at the safe_t boundary.
    The model is copied: nn.Module.double() would mutate the caller's model.
    """
    if y.dtype == torch.float64:
        return model, t, y
    return copy.deepcopy(model).double(), t.double(), y.double()


def bridge_y(y_0, y_T, t, eps):
    """Sample Y_t on the Brownian bridge between Y_0 and Y_T (Eq. 12).

    Args:
        y_0: (n, d) bridge start.
        y_T: (n, d) bridge end.
        t: scalar time in [0, 1).
        eps: diffusion scale.

    Returns:
        (n, d) sample of Y_t.
    """
    t = torch.as_tensor(t, dtype=y_0.dtype, device=y_0.device)
    return y_T * t + y_0 * (1.0 - t) + torch.sqrt(eps * t * (1.0 - t)) * torch.randn_like(y_0)


def map_x(model, t, y, beta):
    """Apply X_t = Y_t + (1/beta) s(t, Y_t), the Bass stretch of the Y-process.

    Args:
        model: trained LightSBM.
        t: (n,) times.
        y: (n, d) points in Y-space.
        beta: SBB volatility penalty.

    Returns:
        (n, d) points in X-space.
    """
    return y + model.get_drift(t, y) / beta


def sigma_x(model, t, y, beta):
    """Evaluate the learned volatility sigma = sqrt(eps) (I + (1/beta) D_y s).

    Args:
        model: trained LightSBM.
        t: (n,) times.
        y: (n, d) points in Y-space.
        beta: SBB volatility penalty.

    Returns:
        (n, d, d) volatility matrices, indexed on the matching X_t.
    """
    dtype = y.dtype
    model64, t64, y64 = _as_double(model, t, y)
    y64 = y64.detach().requires_grad_(True)
    jac = jacobian(score(model64, t64, y64), y64)
    eye = torch.eye(y64.shape[1], dtype=y64.dtype, device=y64.device)
    return (torch.sqrt(model64.epsilon) * (eye + jac / beta)).to(dtype)


def drift_x(model, t, y, beta):
    """Evaluate the learned SBB drift at Y_t, indexed on the matching X_t.

    Ito on X_t = Y_t + (1/beta) s(t, Y_t) with dY = s dt + sqrt(eps) dW gives
    a = s + (1/beta) (d_t s + D_y s . s + (eps/2) lap s).

    Args:
        model: trained LightSBM.
        t: (n,) times.
        y: (n, d) points in Y-space.
        beta: SBB volatility penalty.

    Returns:
        (n, d) drift a(t, X_t).
    """
    dtype = y.dtype
    model64, t64, y64 = _as_double(model, t, y)
    y64 = y64.detach().requires_grad_(True)
    t64 = t64.detach().requires_grad_(True)
    s = score(model64, t64, y64)

    jac = jacobian(s, y64)
    # grad against t must be taken per component: a summed grad_outputs would
    # collapse the d time derivatives into one and broadcast it back.
    dt_s = torch.stack([torch.autograd.grad(s[:, i], t64, grad_outputs=torch.ones_like(s[:, i]),
                                            create_graph=True, retain_graph=True)[0]
                        for i in range(y64.shape[1])], dim=1)
    lap = torch.stack([jacobian(jac[:, i, :], y64).diagonal(dim1=-2, dim2=-1).sum(-1)
                       for i in range(y64.shape[1])], dim=1)

    ito = dt_s + (jac @ s.unsqueeze(-1)).squeeze(-1) + 0.5 * model64.epsilon * lap
    return (s + ito / beta).to(dtype)


def jacobian(f, y):
    """Row-by-row Jacobian of `f` with respect to `y`.

    Args:
        f: (n, k) output built from `y` with create_graph=True.
        y: (n, d) input with requires_grad.

    Returns:
        (n, k, d) with entry [.., i, j] = d f_i / d y_j.
    """
    rows = []
    for i in range(f.shape[1]):
        # allow_unused: a locally affine f has a constant Jacobian, so the second
        # derivative legitimately has no path back to y.
        g = torch.autograd.grad(f[:, i], y, grad_outputs=torch.ones_like(f[:, i]),
                                create_graph=True, retain_graph=True, allow_unused=True)[0]
        rows.append(torch.zeros_like(y) if g is None else g)
    return torch.stack(rows, dim=1)


def encode_y(model, model_inv, x, t, beta, safe_t=1e-2):
    """Map X-space samples into Y-space with the regime-appropriate inverse.

    Args:
        model: trained LightSBM.
        model_inv: MLP inverse net, or None in the large-beta regime.
        x: (n, d) points in X-space.
        t: scalar time.
        beta: SBB volatility penalty.
        safe_t: margin keeping t away from 1.

    Returns:
        (n, d) points in Y-space.
    """
    t_col = torch.full((len(x), 1), clamp_t(t, safe_t), dtype=x.dtype, device=x.device)
    drift = model.get_drift(t_col.squeeze(-1), x)
    if model_inv is None:
        return (x - drift / beta).detach()
    return model_inv(t_col, (x + drift / beta).detach()).detach()


def trajectories(model, y_0, y_T, times, beta):
    """Reconstruct the X_t marginals along the bridge.

    Args:
        model: trained LightSBM.
        y_0: (n, d) bridge start in Y-space.
        y_T: (n, d) bridge end in Y-space.
        times: (m,) times in [0, 1).
        beta: SBB volatility penalty.

    Returns:
        (n, m, d) reconstructed X_t.
    """
    out = []
    for t in times:
        t = clamp_t(t)
        y_t = bridge_y(y_0, y_T, t, model.epsilon)
        t_vec = torch.full((len(y_t),), t, dtype=y_t.dtype, device=y_t.device)
        out.append(map_x(model, t_vec, y_t, beta).detach())
    return torch.stack(out, dim=1)
