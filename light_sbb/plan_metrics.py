"""Metrics comparing a learned SBB plan against the analytic Gaussian reference.

Metric A is the sliced Wasserstein distance on the joint plan, B the cross-covariance
error, C the objective gap, and D the drift and volatility recovery errors.
"""

import numpy as np
import torch

import gaussian_ground_truth as gt
from control_extraction import bridge_y, clamp_t, drift_x, map_x, sigma_x


def sliced_wasserstein(x, y, n_projections=512, seed=0):
    """Sliced 2-Wasserstein distance between two point clouds (metric A).

    Args:
        x: (n, d) samples.
        y: (m, d) samples.
        n_projections: number of random directions.
        seed: RNG seed for the projections.

    Returns:
        Scalar distance.
    """
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=(x.shape[1], n_projections))
    theta /= np.linalg.norm(theta, axis=0, keepdims=True)

    px = np.sort(np.asarray(x) @ theta, axis=0)
    py = np.sort(np.asarray(y) @ theta, axis=0)
    if px.shape[0] != py.shape[0]:
        q = np.linspace(0.0, 1.0, min(px.shape[0], py.shape[0]))
        px = np.quantile(px, q, axis=0)
        py = np.quantile(py, q, axis=0)
    return float(np.sqrt(((px - py) ** 2).mean()))


def cross_covariance_error(pairs, beta, sigma2=(10.0, 0.1)):
    """Relative Frobenius error on the joint covariance (metric B).

    Args:
        pairs: (n, 2d) samples of (X_0, X_1).
        beta: SBB volatility penalty.
        sigma2: target marginal variances.

    Returns:
        Scalar relative error.
    """
    exact = gt.solution(beta, sigma2)["cov_joint"]
    empirical = np.cov(np.asarray(pairs).T)
    return float(np.linalg.norm(empirical - exact) / np.linalg.norm(exact))


def objective(model, y_0, y_T, beta, times, eps=1.0):
    """Estimate the SBB objective of a learned model (metric C, numerator).

    Computes E[int_0^1 (||a||^2 + beta ||sigma - sqrt(eps) I||_F^2) dt] by sampling
    the Y-bridge at each time and evaluating both controls there.

    Args:
        model: trained LightSBM.
        y_0: (n, d) bridge start in Y-space.
        y_T: (n, d) bridge end in Y-space.
        beta: SBB volatility penalty.
        times: (m,) quadrature times in [0, 1).
        eps: diffusion scale.

    Returns:
        Scalar objective estimate.
    """
    d = y_0.shape[1]
    eye = torch.eye(d, dtype=y_0.dtype, device=y_0.device)
    ts, vals = [], []
    for t in times:
        t = clamp_t(t)
        y_t = bridge_y(y_0, y_T, t, torch.as_tensor(eps, dtype=y_0.dtype, device=y_0.device))
        t_vec = torch.full((len(y_t),), t, dtype=y_t.dtype, device=y_t.device)
        a = drift_x(model, t_vec, y_t, beta).detach()
        sigma = sigma_x(model, t_vec, y_t, beta).detach()
        drift_cost = (a**2).sum(-1)
        vol_cost = beta * ((sigma - np.sqrt(eps) * eye) ** 2).sum((-2, -1))
        ts.append(t)
        vals.append(float((drift_cost + vol_cost).mean()))
    return float(np.trapz(vals, ts))


def objective_gap(model, y_0, y_T, beta, times, sigma2=(10.0, 0.1), eps=1.0):
    """Relative objective gap (J_hat - J*) / |J*| (metric C).

    J* is evaluated on the same clamped grid as J_hat: comparing against the exact
    J* over [0, 1] would charge the model for the tail that safe_t truncates.
    """
    clamped = [clamp_t(t) for t in times]
    j_star = gt.optimal_objective(gt.solution(beta, sigma2)["r"], beta, times=clamped)
    return (objective(model, y_0, y_T, beta, times, eps) - j_star) / abs(j_star)


def drift_only_gap(model, y_0, y_T, beta, times, sigma2=(10.0, 0.1), eps=1.0):
    """Objective gap counting the drift cost alone, on both sides.

    The SB baseline pays no volatility cost, so its `objective_gap` is scored against
    an inflated reference. This column drops that term everywhere, making the two
    methods comparable on the drift alone.
    """
    ts, vals = [], []
    for t in times:
        t = clamp_t(t)
        y_t = bridge_y(y_0, y_T, t, torch.as_tensor(eps, dtype=y_0.dtype, device=y_0.device))
        t_vec = torch.full((len(y_t),), t, dtype=y_t.dtype, device=y_t.device)
        a = drift_x(model, t_vec, y_t, beta).detach()
        ts.append(t)
        vals.append(float((a**2).sum(-1).mean()))

    grid = np.asarray(ts, dtype=float)
    j_star = sum(np.trapz(gt.drift_coeff(grid, r) ** 2 * gt.variance_path(grid, r, beta), grid)
                 for r in gt.solution(beta, sigma2)["r"])
    return (float(np.trapz(vals, ts)) - j_star) / abs(j_star)


def control_errors(model, y_0, y_T, beta, times, sigma2=(10.0, 0.1), eps=1.0):
    """Relative recovery errors for the drift and the volatility (metric D).

    The note indexes E_a on the exact states X*_t, but the learned drift is only
    computable at X_t = map_x(y_t): grading it elsewhere would need the inverse of
    the map, which this construction exists to avoid. So the learned and exact
    controls are compared at the model's own reconstructed states, while both
    denominators are the exact E||a*(t, X*_t)||^2 and ||b*(t)||_F^2. A model whose
    map is wrong is therefore penalised through the numerator, not excused by a
    matching denominator.

    Args:
        model: trained LightSBM.
        y_0: (n, d) bridge start in Y-space.
        y_T: (n, d) bridge end in Y-space.
        beta: SBB volatility penalty.
        times: (m,) quadrature times in [0, 1).
        sigma2: target marginal variances.
        eps: diffusion scale.

    Returns:
        dict with keys `E_a` and `E_b`.
    """
    r = gt.solution(beta, sigma2)["r"]
    ts, a_num, a_den, b_num, b_den = [], [], [], [], []

    for t in times:
        t = clamp_t(t)
        y_t = bridge_y(y_0, y_T, t, torch.as_tensor(eps, dtype=y_0.dtype, device=y_0.device))
        t_vec = torch.full((len(y_t),), t, dtype=y_t.dtype, device=y_t.device)
        x_t = map_x(model, t_vec, y_t, beta).detach()

        a_hat = drift_x(model, t_vec, y_t, beta).detach()
        sigma_hat = sigma_x(model, t_vec, y_t, beta).detach()

        coeff = torch.tensor([gt.drift_coeff(t, ri) for ri in r],
                             dtype=x_t.dtype, device=x_t.device)
        b_star = torch.diag(torch.tensor([gt.vol_coeff(t, ri, beta) for ri in r],
                                         dtype=x_t.dtype, device=x_t.device))

        ts.append(t)
        a_num.append(float(((a_hat - coeff * x_t) ** 2).sum(-1).mean()))
        a_den.append(sum(gt.drift_coeff(t, ri) ** 2 * gt.variance_path(t, ri, beta)[0]
                         for ri in r))
        b_num.append(float(((sigma_hat - b_star) ** 2).sum((-2, -1)).mean()))
        b_den.append(float((b_star**2).sum()))

    return {"E_a": float(np.trapz(a_num, ts) / np.trapz(a_den, ts)),
            "E_b": float(np.trapz(b_num, ts) / np.trapz(b_den, ts))}


def terminal_wasserstein(x_hat, x_star):
    """Sliced 2-Wasserstein on the terminal marginal, for the paper's first column."""
    return sliced_wasserstein(x_hat, x_star)


def evaluate(model, pairs, y_0, y_T, beta, n_times=21, sigma2=(10.0, 0.1), eps=1.0, seed=0):
    """Run metrics A-D for one trained model.

    Args:
        model: trained LightSBM.
        pairs: (n, 2d) generated (X_0, X_hat_1) samples.
        y_0: (n, d) bridge start in Y-space.
        y_T: (n, d) bridge end in Y-space.
        beta: SBB volatility penalty.
        n_times: quadrature nodes over [0, 1).
        sigma2: target marginal variances.
        eps: diffusion scale.
        seed: RNG seed for the reference sample and the projections.

    Returns:
        dict with `plan_sw2`, `terminal_sw2`, `cross_cov_err`, `objective_gap`,
        `E_a` and `E_b`.
    """
    rng = np.random.default_rng(seed)
    exact = gt.sample_plan(beta, len(pairs), sigma2, rng=rng)
    pairs = np.asarray(pairs)
    times = np.linspace(0.0, 1.0, n_times)
    d = y_0.shape[1]

    out = {"plan_sw2": sliced_wasserstein(pairs, exact, seed=seed),
           "terminal_sw2": terminal_wasserstein(pairs[:, d:], exact[:, d:]),
           "cross_cov_err": cross_covariance_error(pairs, beta, sigma2),
           "objective_gap": objective_gap(model, y_0, y_T, beta, times, sigma2, eps),
           "drift_only_gap": drift_only_gap(model, y_0, y_T, beta, times, sigma2, eps)}
    out.update(control_errors(model, y_0, y_T, beta, times, sigma2, eps))
    return out
