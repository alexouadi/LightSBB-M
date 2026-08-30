"""Analytic Gaussian Schrodinger-Bass bridge (Hasenbichler, Pammer & Thonhauser 2026, Ex. 3.4).

Reference solution for the 2D ground-truth experiment: per-coordinate controls, the exact
joint plan, and the optimal SBB objective.
"""

import numpy as np

EPS = 1.0
_BISECTION_TOL = 1e-14
_BISECTION_MAX_ITER = 400
_VAR_GRID = 200001


def sigma_of_r(r, beta):
    """Evaluate the target variance implied by a root candidate `r`.

    Args:
        r: scalar in (-inf, beta/(beta+1)), the admissible branch.
        beta: SBB volatility penalty.

    Returns:
        sigma^2 = 1/(1-r)^2 + beta^2 / [(beta-r)(beta-(beta+1)r)].
    """
    return 1.0 / (1.0 - r) ** 2 + beta**2 / ((beta - r) * (beta - (beta + 1.0) * r))


def solve_r(sigma2, beta):
    """Solve for the admissible root r < beta/(beta+1) matching a target variance.

    Args:
        sigma2: target marginal variance of the coordinate.
        beta: SBB volatility penalty.

    Returns:
        The unique admissible root as a float.
    """
    upper = beta / (beta + 1.0)
    # sigma_of_r increases monotonically on the admissible branch, from 0 as r -> -inf
    # to +inf at r = beta/(beta+1), so a plain bisection is well posed.
    hi = upper * (1.0 - 1e-13)
    lo = -1.0
    while sigma_of_r(lo, beta) > sigma2:
        lo *= 2.0
        if lo < -1e12:
            raise ValueError(f"no admissible root bracketed for sigma2={sigma2}, beta={beta}")

    for _ in range(_BISECTION_MAX_ITER):
        mid = 0.5 * (lo + hi)
        if sigma_of_r(mid, beta) < sigma2:
            lo = mid
        else:
            hi = mid
        if hi - lo < _BISECTION_TOL * max(1.0, abs(mid)):
            break
    return 0.5 * (lo + hi)


def drift_coeff(t, r):
    """Exact drift coefficient a^beta_i(t) of dX_t = a(t) X_t dt + b(t) dB_t."""
    return r / (1.0 - (1.0 - t) * r)


def vol_coeff(t, r, beta):
    """Exact volatility coefficient b^beta_i(t)."""
    num = beta * (1.0 - (1.0 - t) * r)
    return num / (num - r)


def cross_cov(r):
    """Exact Cov(X_0^i, X_1^i) = 1 / (1 - r)."""
    return 1.0 / (1.0 - r)


def _phi(t, r):
    """Integrating factor exp(int_0^t a(s) ds) = (1 - (1-t) r) / (1 - r)."""
    return (1.0 - (1.0 - t) * r) / (1.0 - r)


def variance_path(t, r, beta):
    """Variance of X_t^i along the exact bridge.

    Solves dV/dt = 2 a(t) V + b(t)^2 with V(0) = 1 by variation of constants,
    V(t) = phi(t)^2 [1 + int_0^t (b(s)/phi(s))^2 ds], phi(t) = exp(int_0^t a).

    Args:
        t: (m,) times in [0, 1].
        r: admissible root for the coordinate.
        beta: SBB volatility penalty.

    Returns:
        (m,) variances. V(1) equals the coordinate's target variance.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    phi = _phi(t, r)
    grid = np.linspace(0.0, 1.0, _VAR_GRID)
    integrand = (vol_coeff(grid, r, beta) / _phi(grid, r)) ** 2
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(grid))])
    return phi**2 * (1.0 + np.interp(t, grid, cum))


def optimal_objective(betas_r, beta, n_quad=200001, times=None):
    """Compute J* = sum_i int_0^1 ( a_i(t)^2 E[(X_t^i)^2] + beta (b_i(t) - sqrt(eps))^2 ) dt.

    Args:
        betas_r: sequence of admissible roots, one per coordinate.
        beta: SBB volatility penalty.
        n_quad: number of trapezoid nodes, when `times` is not given.
        times: optional explicit quadrature grid. Pass the same nodes used to
            estimate a learned objective, otherwise the two differ by the tail
            that a safe_t-truncated grid drops (about 2% at safe_t = 1e-2).

    Returns:
        The scalar objective value.
    """
    grid = np.linspace(0.0, 1.0, n_quad) if times is None else np.asarray(times, dtype=float)
    total = 0.0
    for r in betas_r:
        a = drift_coeff(grid, r)
        b = vol_coeff(grid, r, beta)
        v = variance_path(grid, r, beta)
        integrand = a**2 * v + beta * (b - np.sqrt(EPS)) ** 2
        total += np.trapz(integrand, grid)
    return total


def solution(beta, sigma2=(10.0, 0.1)):
    """Assemble the full analytic solution for one beta.

    Args:
        beta: SBB volatility penalty.
        sigma2: target marginal variances, one per coordinate.

    Returns:
        dict with `r` (d,), `cross_cov` (d,), `cov_joint` (2d, 2d) and `J_star`.
    """
    r = np.array([solve_r(s, beta) for s in sigma2])
    c = cross_cov(r)
    d = len(r)
    cov = np.zeros((2 * d, 2 * d))
    cov[:d, :d] = np.eye(d)
    cov[d:, d:] = np.diag(sigma2)
    cov[:d, d:] = np.diag(c)
    cov[d:, :d] = np.diag(c)
    return {"beta": beta, "r": r, "cross_cov": c, "cov_joint": cov,
            "J_star": optimal_objective(r, beta)}


def sample_plan(beta, n, sigma2=(10.0, 0.1), rng=None):
    """Sample (X_0, X_1) pairs from the exact optimal plan.

    Args:
        beta: SBB volatility penalty.
        n: number of pairs.
        sigma2: target marginal variances.
        rng: optional numpy Generator.

    Returns:
        (n, 2d) array, columns [X_0, X_1].
    """
    rng = np.random.default_rng() if rng is None else rng
    cov = solution(beta, sigma2)["cov_joint"]
    return rng.multivariate_normal(np.zeros(cov.shape[0]), cov, size=n)


if __name__ == "__main__":
    print(f"{'beta':>5} {'r1':>10} {'r2':>10} {'cov1':>9} {'cov2':>9} {'J*':>12}")
    for beta in (2.0, 10.0, 100.0):
        sol = solution(beta)
        print(f"{beta:>5g} {sol['r'][0]:>10.4f} {sol['r'][1]:>10.4f}"
              f" {sol['cross_cov'][0]:>9.4f} {sol['cross_cov'][1]:>9.4f}"
              f" {sol['J_star']:>12.4f}")
