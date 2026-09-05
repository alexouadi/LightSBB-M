"""Quantitative fit of a generated terminal marginal against the true target.

Mirrors the heavy-tail table of the paper (W1, energy distance, KS) on the metrics
that transfer to a low-entropy target, and adds the moment and quantile diagnostics
that measure concentration, which is what the separation tests vary.
"""

import numpy as np

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _flat(x):
    """Return a 1D float array, projecting a (n, 1) column if needed."""
    return np.asarray(x, dtype=float).ravel()


def wasserstein_1d(x, y):
    """Exact 1-Wasserstein distance between two 1D samples.

    Args:
        x: (n,) or (n, 1) generated samples.
        y: (m,) or (m, 1) target samples.

    Returns:
        Scalar distance.
    """
    x, y = np.sort(_flat(x)), np.sort(_flat(y))
    if len(x) != len(y):
        q = np.linspace(0.0, 1.0, min(len(x), len(y)))
        x, y = np.quantile(x, q), np.quantile(y, q)
    return float(np.abs(x - y).mean())


def energy_distance(x, y, n_max=4000, seed=0):
    """Energy distance 2 E|X-Y| - E|X-X'| - E|Y-Y'| between two 1D samples.

    Args:
        x: (n,) or (n, 1) generated samples.
        y: (m,) or (m, 1) target samples.
        n_max: subsample size, since the statistic is quadratic in n.
        seed: RNG seed for the subsampling.

    Returns:
        Scalar distance.
    """
    rng = np.random.default_rng(seed)
    x, y = _flat(x), _flat(y)
    if len(x) > n_max:
        x = rng.choice(x, n_max, replace=False)
    if len(y) > n_max:
        y = rng.choice(y, n_max, replace=False)
    d_xy = np.abs(x[:, None] - y[None, :]).mean()
    d_xx = np.abs(x[:, None] - x[None, :]).mean()
    d_yy = np.abs(y[:, None] - y[None, :]).mean()
    return float(2 * d_xy - d_xx - d_yy)


def ks_statistic(x, y):
    """Two-sample Kolmogorov-Smirnov statistic, the sup gap between empirical CDFs.

    Args:
        x: (n,) or (n, 1) generated samples.
        y: (m,) or (m, 1) target samples.

    Returns:
        Scalar statistic in [0, 1].
    """
    x, y = np.sort(_flat(x)), np.sort(_flat(y))
    grid = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, grid, side="right") / len(x)
    cdf_y = np.searchsorted(y, grid, side="right") / len(y)
    return float(np.abs(cdf_x - cdf_y).max())


def quantile_errors(x, y, quantiles=QUANTILES):
    """Absolute error at a set of quantiles, plus the worst of them.

    Args:
        x: (n,) or (n, 1) generated samples.
        y: (m,) or (m, 1) target samples.
        quantiles: probabilities at which to compare.

    Returns:
        dict with one `q<p>_err` entry per quantile and their maximum `quantile_err_max`.
    """
    qx = np.quantile(_flat(x), quantiles)
    qy = np.quantile(_flat(y), quantiles)
    out = {f"q{int(100 * q):02d}_err": float(abs(a - b))
           for q, a, b in zip(quantiles, qx, qy)}
    out["quantile_err_max"] = float(np.abs(qx - qy).max())
    return out


def moment_errors(x, y):
    """Relative errors on the first two moments of the marginal.

    The standard-deviation error is the decisive one for these tests: a Schrodinger
    bridge with frozen diffusion cannot contract below sqrt(eps) without paying an
    unbounded drift cost, so it should saturate while the target keeps shrinking.

    Args:
        x: (n,) or (n, 1) generated samples.
        y: (m,) or (m, 1) target samples.

    Returns:
        dict with `mean_err`, `std_gen`, `std_true` and `std_rel_err`.
    """
    x, y = _flat(x), _flat(y)
    std_true = float(y.std())
    return {"mean_err": float(abs(x.mean() - y.mean())),
            "std_gen": float(x.std()),
            "std_true": std_true,
            "std_rel_err": float(abs(x.std() - std_true) / std_true)}


def evaluate(x, y, seed=0, prefix=""):
    """Run every distribution metric on one generated/target pair of 1D samples.

    Args:
        x: (n,) or (n, 1) generated samples.
        y: (m,) or (m, 1) target samples.
        seed: RNG seed for the energy-distance subsampling.
        prefix: string prepended to every key, to distinguish coordinates.

    Returns:
        dict of scalar metrics.
    """
    out = {"w1": wasserstein_1d(x, y),
           "energy": energy_distance(x, y, seed=seed),
           "ks": ks_statistic(x, y)}
    out.update(moment_errors(x, y))
    out.update(quantile_errors(x, y))
    return {f"{prefix}{k}": v for k, v in out.items()}


def evaluate_per_coordinate(x, y, seed=0, names=None):
    """Run the distribution metrics on each coordinate of a multivariate sample.

    Args:
        x: (n, d) generated samples.
        y: (m, d) target samples.
        seed: RNG seed for the energy-distance subsampling.
        names: optional per-coordinate key prefixes; defaults to `x0_`, `x1_`, ...

    Returns:
        dict of scalar metrics, one block per coordinate.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.atleast_2d(np.asarray(y, dtype=float))
    names = names or [f"x{i}_" for i in range(x.shape[1])]
    out = {}
    for i, name in enumerate(names):
        out.update(evaluate(x[:, i], y[:, i], seed=seed, prefix=name))
    return out
