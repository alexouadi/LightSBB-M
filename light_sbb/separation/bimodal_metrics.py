"""Metrics for Test B, the 1D bimodal target that has no closed-form reference.

Without an analytic optimum the model is graded on how well it hits the target
marginal, what it actually pays, and how far its learned volatility varies with the
state, which is the property the note proves must hold at the optimum.
"""

import numpy as np
import torch

import distribution_metrics as dm
from control_extraction import bridge_y, clamp_t, drift_x, map_x, sigma_x

MODE = 2.0


def mode_statistics(x, mode=MODE):
    """Per-mode occupancy and spread of a bimodal sample.

    Args:
        x: (n,) or (n, 1) generated terminal samples.
        mode: absolute location of the two target modes.

    Returns:
        dict with the positive-mode mass, the two per-mode means and stds, and the
        worst absolute error on the mode locations. A mode the sample never reaches
        is charged the full distance `mode`, so a collapse is penalised rather than
        dropped by a NaN that downstream averaging would silently skip.
    """
    x = np.asarray(x, dtype=float).ravel()
    right = x >= 0.0
    out = {"mass_right": float(right.mean())}
    errors = []
    for name, sel in (("neg", ~right), ("pos", right)):
        part = x[sel]
        empty = len(part) == 0
        out[f"mean_{name}"] = float("nan") if empty else float(part.mean())
        out[f"std_{name}"] = float("nan") if empty else float(part.std())
        errors.append(mode if empty else abs(abs(out[f"mean_{name}"]) - mode))
    out["mode_err"] = float(max(errors))
    return out


def achieved_cost(model, y_0, y_T, beta, times, eps=1.0, sb_baseline=False):
    """Estimate the drift and volatility costs the model actually pays.

    Args:
        model: trained LightSBM.
        y_0: (n, d) bridge start in Y-space.
        y_T: (n, d) bridge end in Y-space.
        beta: volatility penalty.
        times: (m,) quadrature times in [0, 1).
        eps: diffusion scale.
        sb_baseline: when True the model is a Schrodinger bridge, whose volatility is
            sqrt(eps) I by construction, so none is extracted and its cost is zero.

    Returns:
        dict with `drift_cost`, `vol_cost` and their sum `total_cost`.
    """
    d = y_0.shape[1]
    eye = torch.eye(d, dtype=y_0.dtype, device=y_0.device)
    ts, drift, vol = [], [], []

    for t in times:
        t = clamp_t(t)
        y_t = bridge_y(y_0, y_T, t, torch.as_tensor(eps, dtype=y_0.dtype, device=y_0.device))
        t_vec = torch.full((len(y_t),), t, dtype=y_t.dtype, device=y_t.device)

        if sb_baseline:
            a = model.get_drift(t_vec, y_t).detach()
            vol.append(0.0)
        else:
            a = drift_x(model, t_vec, y_t, beta).detach()
            sigma = sigma_x(model, t_vec, y_t, beta).detach()
            vol.append(float((beta * ((sigma - np.sqrt(eps) * eye) ** 2).sum((-2, -1))).mean()))
        ts.append(t)
        drift.append(float((a**2).sum(-1).mean()))

    drift_cost = float(np.trapz(drift, ts))
    vol_cost = float(np.trapz(vol, ts))
    return {"drift_cost": drift_cost, "vol_cost": vol_cost,
            "total_cost": drift_cost + vol_cost}


def volatility_profile(model, beta, times, grid, eps=1.0, device="cpu"):
    """Evaluate the learned scalar volatility along the bridge the model actually uses.

    The note predicts sigma* is genuinely state-dependent for Test B, small near the
    modes and larger in the basin between them. sigma_x takes Y-space points, so the
    grid is quantile-matched to the model's own Y_T rather than laid out in X-space:
    a fixed X grid would be read as Y and sample sigma away from the target's support.

    Args:
        model: trained LightSBM.
        beta: volatility penalty.
        times: (m,) quadrature times in [0, 1).
        grid: (g,) points in Y-space, built by `y_grid_from_bridge`.
        eps: diffusion scale.
        device: torch device.

    Returns:
        dict with `times` (m,), `grid` (g,) in Y-space, `grid_x` (m, g) holding the
        X-space point each sigma is indexed on, `sigma` (m, g), and `sigma_spread`,
        the time-averaged range of sigma across the grid over sqrt(eps).
    """
    grid_t = torch.tensor(np.asarray(grid, dtype=np.float32).reshape(-1, 1), device=device)
    rows, xs, ts = [], [], []
    for t in times:
        t = clamp_t(t)
        t_vec = torch.full((len(grid_t),), t, dtype=grid_t.dtype, device=device)
        sigma = sigma_x(model, t_vec, grid_t, beta).detach().reshape(-1)
        rows.append(sigma.cpu().numpy())
        # sigma is evaluated at Y but indexed on the matching X, so the plotting
        # abscissa is the image of the grid under the Bass map, not the grid itself.
        xs.append(map_x(model, t_vec, grid_t, beta).detach().reshape(-1).cpu().numpy())
        ts.append(t)

    sigma = np.stack(rows)
    spread = float(np.mean(sigma.max(axis=1) - sigma.min(axis=1)) / np.sqrt(eps))
    return {"times": ts, "grid": np.asarray(grid, dtype=float).tolist(),
            "grid_x": np.stack(xs).tolist(),
            "sigma": sigma.tolist(), "sigma_spread": spread}


def y_grid_from_bridge(y_T, n_points=81, pad=0.1):
    """Build a Y-space grid spanning the terminal states the model actually reaches.

    Args:
        y_T: (n, 1) terminal Y-space states of the trained bridge.
        n_points: number of grid points.
        pad: fraction of the range added on each side.

    Returns:
        (n_points,) numpy grid covering [min, max] of y_T, padded.
    """
    lo, hi = float(y_T.min()), float(y_T.max())
    margin = pad * (hi - lo)
    return np.linspace(lo - margin, hi + margin, n_points)


def evaluate(model, pairs, y_0, y_T, beta, target_sample, n_times=21, eps=1.0,
             seed=0, sb_baseline=False, grid=None, device="cpu"):
    """Score one trained model on Test B against an empirical target sample.

    Args:
        model: trained LightSBM.
        pairs: (n, 2) generated (X_0, X_hat_1) samples.
        y_0: (n, 1) bridge start in Y-space.
        y_T: (n, 1) bridge end in Y-space.
        beta: volatility penalty.
        target_sample: (n, 1) samples drawn from the true target.
        n_times: quadrature nodes over [0, 1).
        eps: diffusion scale.
        seed: RNG seed for the subsampling inside the distribution metrics.
        sb_baseline: whether the model is the Schrodinger-bridge baseline.
        grid: optional (g,) Y-space grid for the volatility profile; when omitted it
            is derived from `y_T` so it spans the states the bridge actually reaches.
        device: torch device.

    Returns:
        dict of scalar metrics, plus `vol_profile` when `grid` is given and the model
        is not the SB baseline.
    """
    pairs = np.asarray(pairs)
    times = np.linspace(0.0, 1.0, n_times)
    x_1 = pairs[:, 1:]

    out = dm.evaluate(x_1, target_sample, seed=seed)
    out.update(mode_statistics(x_1))
    out.update(achieved_cost(model, y_0, y_T, beta, times, eps, sb_baseline))

    if sb_baseline:
        # A Schrodinger bridge has sigma = sqrt(eps) I everywhere by construction.
        out["sigma_spread"] = 0.0
    else:
        grid = y_grid_from_bridge(y_T) if grid is None else grid
        profile = volatility_profile(model, beta, times, grid, eps, device)
        out["sigma_spread"] = profile.pop("sigma_spread")
        out["vol_profile"] = profile
    return out
