"""LightSB-M baseline on the analytic Gaussian benchmark.

LightSB-M solves the Schrodinger bridge, so it has no volatility penalty and no Bass
map: its volatility is sigma = sqrt(eps) I by construction. Training is the large-beta
LightSBB loop at K = 1, whose k == 0 branch never uses beta, and the metrics that need
a volatility use the fixed sqrt(eps) I instead of extracting one from the model.
"""

import numpy as np
import torch

import gaussian_ground_truth as gt
import plan_metrics as pm
from control_extraction import bridge_y, clamp_t


def sb_drift(model, t, y):
    """Evaluate the SB drift, which is the LightSB-M control itself (no Bass correction).

    Args:
        model: trained LightSBM.
        t: (n,) times in [0, 1).
        y: (n, d) states.

    Returns:
        (n, d) drift.
    """
    return model.get_drift(t, y).detach()


def objective(model, y_0, y_T, beta, times, eps=1.0):
    """SBB objective of the SB baseline, with sigma pinned to sqrt(eps) I.

    The volatility term beta ||sigma - sqrt(eps) I||^2 vanishes identically, so the
    baseline only pays the drift cost. Beta is still the column's beta: the point is
    to score LightSB-M on the SBB objective it does not optimize.

    Args:
        model: trained LightSBM.
        y_0: (n, d) bridge start.
        y_T: (n, d) bridge end.
        beta: SBB volatility penalty of the column being scored.
        times: (m,) quadrature times in [0, 1).
        eps: diffusion scale.

    Returns:
        Scalar objective estimate.
    """
    ts, vals = [], []
    for t in times:
        t = clamp_t(t)
        y_t = bridge_y(y_0, y_T, t, torch.as_tensor(eps, dtype=y_0.dtype, device=y_0.device))
        t_vec = torch.full((len(y_t),), t, dtype=y_t.dtype, device=y_t.device)
        a = sb_drift(model, t_vec, y_t)
        ts.append(t)
        vals.append(float((a**2).sum(-1).mean()))
    return float(np.trapz(vals, ts))


def objective_gap(model, y_0, y_T, beta, times, sigma2=(10.0, 0.1), eps=1.0):
    """Relative objective gap of the baseline against the same clamped J*.

    The gap can come out negative, and that does not mean the baseline beats
    LightSBB-M: its J_hat omits the volatility term that J* charges the true optimum,
    so it is compared against an inflated reference. Report it alongside
    `drift_only_gap`, which removes that asymmetry.
    """
    clamped = [clamp_t(t) for t in times]
    j_star = gt.optimal_objective(gt.solution(beta, sigma2)["r"], beta, times=clamped)
    return (objective(model, y_0, y_T, beta, times, eps) - j_star) / abs(j_star)


def drift_only_objective(betas_r, beta, times):
    """Reference cost of the exact optimum with its volatility term removed.

    Args:
        betas_r: admissible roots, one per coordinate.
        beta: SBB volatility penalty.
        times: quadrature grid, already clamped.

    Returns:
        Scalar int sum_i a_i(t)^2 Var(X_t^i) dt.
    """
    grid = np.asarray(times, dtype=float)
    return sum(np.trapz(gt.drift_coeff(grid, r) ** 2 * gt.variance_path(grid, r, beta), grid)
               for r in betas_r)


def drift_only_gap(model, y_0, y_T, beta, times, sigma2=(10.0, 0.1), eps=1.0):
    """Objective gap with the volatility term dropped on both sides.

    This is the comparison `objective_gap` cannot make fairly: both the baseline and
    the reference are charged for their drift alone, so a positive value really does
    mean a worse drift rather than a missing cost term.
    """
    clamped = [clamp_t(t) for t in times]
    j_star = drift_only_objective(gt.solution(beta, sigma2)["r"], beta, clamped)
    return (objective(model, y_0, y_T, beta, times, eps) - j_star) / abs(j_star)


def control_errors(model, y_0, y_T, beta, times, sigma2=(10.0, 0.1), eps=1.0):
    """Drift and volatility recovery of the baseline (metric D).

    E_b needs no model: sigma is identically sqrt(eps) I, so the error against the
    exact b*(t) is a closed-form number quantifying what a Schrodinger bridge cannot
    represent. E_a compares the SB drift to the exact one at the SB's own states,
    which for this model are the Y-states themselves (the Bass map is the identity).

    Args:
        model: trained LightSBM.
        y_0: (n, d) bridge start.
        y_T: (n, d) bridge end.
        beta: SBB volatility penalty of the column being scored.
        times: (m,) quadrature times in [0, 1).
        sigma2: target marginal variances.
        eps: diffusion scale.

    Returns:
        dict with keys `E_a` and `E_b`.
    """
    r = gt.solution(beta, sigma2)["r"]
    d = y_0.shape[1]
    eye = torch.eye(d, dtype=y_0.dtype, device=y_0.device)
    ts, a_num, a_den, b_num, b_den = [], [], [], [], []

    for t in times:
        t = clamp_t(t)
        y_t = bridge_y(y_0, y_T, t, torch.as_tensor(eps, dtype=y_0.dtype, device=y_0.device))
        t_vec = torch.full((len(y_t),), t, dtype=y_t.dtype, device=y_t.device)
        a_hat = sb_drift(model, t_vec, y_t)

        coeff = torch.tensor([gt.drift_coeff(t, ri) for ri in r],
                             dtype=y_t.dtype, device=y_t.device)
        b_star = torch.diag(torch.tensor([gt.vol_coeff(t, ri, beta) for ri in r],
                                         dtype=y_t.dtype, device=y_t.device))

        ts.append(t)
        a_num.append(float(((a_hat - coeff * y_t) ** 2).sum(-1).mean()))
        a_den.append(sum(gt.drift_coeff(t, ri) ** 2 * gt.variance_path(t, ri, beta)[0]
                         for ri in r))
        b_num.append(float(((np.sqrt(eps) * eye - b_star) ** 2).sum()))
        b_den.append(float((b_star**2).sum()))

    return {"E_a": float(np.trapz(a_num, ts) / np.trapz(a_den, ts)),
            "E_b": float(np.trapz(b_num, ts) / np.trapz(b_den, ts))}


def evaluate(model, pairs, y_0, y_T, beta, n_times=21, sigma2=(10.0, 0.1), eps=1.0, seed=0):
    """Run metrics A-D for the LightSB-M baseline, scored against the SBB reference.

    Args:
        model: trained LightSBM.
        pairs: (n, 2d) generated (X_0, X_hat_1) samples.
        y_0: (n, d) bridge start.
        y_T: (n, d) bridge end.
        beta: SBB volatility penalty of the column being scored.
        n_times: quadrature nodes over [0, 1).
        sigma2: target marginal variances.
        eps: diffusion scale.
        seed: RNG seed for the reference sample and the projections.

    Returns:
        dict with the same six keys as `plan_metrics.evaluate`.
    """
    rng = np.random.default_rng(seed)
    exact = gt.sample_plan(beta, len(pairs), sigma2, rng=rng)
    pairs = np.asarray(pairs)
    times = np.linspace(0.0, 1.0, n_times)
    d = y_0.shape[1]

    out = {"plan_sw2": pm.sliced_wasserstein(pairs, exact, seed=seed),
           "terminal_sw2": pm.sliced_wasserstein(pairs[:, d:], exact[:, d:], seed=seed),
           "cross_cov_err": pm.cross_covariance_error(pairs, beta, sigma2),
           "objective_gap": objective_gap(model, y_0, y_T, beta, times, sigma2, eps),
           "drift_only_gap": drift_only_gap(model, y_0, y_T, beta, times, sigma2, eps)}
    out.update(control_errors(model, y_0, y_T, beta, times, sigma2, eps))
    return out
