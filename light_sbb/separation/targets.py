"""Low-entropy targets for the SB/SBB separation tests, and the SB lower bound they trip.

Test A collapses a Gaussian along one axis, Test B concentrates a symmetric bimodal
mixture. Both keep W2 bounded while H(mu_T) -> -infinity as delta -> 0, which is the
regime Corollary 6 of the note separates.
"""

import numpy as np
import torch

DELTAS = (0.8, 0.5, 0.2, 0.1, 0.05, 0.02)
MODE = 2.0


class GaussianSampler:
    """Sample from a centered Gaussian with diagonal covariance."""

    def __init__(self, variances, device="cpu"):
        """Store the per-coordinate standard deviations.

        Args:
            variances: (d,) marginal variances.
            device: torch device the samples are drawn on.
        """
        self.std = torch.tensor(variances, dtype=torch.float32, device=device).sqrt()
        self.device = device

    def sample(self, n, generator=None):
        """Return (n, d) samples on the sampler's device.

        Args:
            n: number of samples.
            generator: optional torch.Generator making the draw reproducible.
        """
        noise = torch.randn(n, len(self.std), device=self.device, generator=generator)
        return noise * self.std


class BimodalSampler:
    """Sample from the 1D mixture 0.5 N(-mode, delta^2) + 0.5 N(+mode, delta^2)."""

    def __init__(self, delta, mode=MODE, device="cpu"):
        """Store the mixture parameters.

        Args:
            delta: common standard deviation of the two components.
            mode: absolute location of the two centers.
            device: torch device the samples are drawn on.
        """
        self.delta = float(delta)
        self.mode = float(mode)
        self.device = device

    def sample(self, n, generator=None):
        """Return (n, 1) samples on the sampler's device.

        Args:
            n: number of samples.
            generator: optional torch.Generator making the draw reproducible.
        """
        signs = torch.randint(0, 2, (n, 1), device=self.device, dtype=torch.float32,
                              generator=generator) * 2.0 - 1.0
        noise = torch.randn(n, 1, device=self.device, generator=generator)
        return signs * self.mode + self.delta * noise


def target_sampler(test, delta, device="cpu"):
    """Build the target sampler for one test at one delta.

    Args:
        test: "A" for the 2D collapsing Gaussian, "B" for the 1D bimodal mixture.
        delta: concentration parameter.
        device: torch device.

    Returns:
        A sampler exposing `sample(n)`.
    """
    if test == "A":
        return GaussianSampler(sigma2_of(delta), device)
    return BimodalSampler(delta, device=device)


def source_sampler(test, device="cpu"):
    """Build the source sampler mu_0, standard normal in the test's dimension."""
    return GaussianSampler([1.0] * dim_of(test), device)


def dim_of(test):
    """Return the state dimension of a test."""
    return 2 if test == "A" else 1


def sigma2_of(delta):
    """Return the Test A target variances diag(1, delta^2)."""
    return [1.0, float(delta) ** 2]


def entropy(test, delta):
    """Differential entropy of the target, exactly for A and as the mixture bound for B.

    For Test B the true entropy has no closed form, so this returns the standard upper
    bound H(sum w_i p_i) <= H(w) + sum w_i H(p_i), which is what the note's divergence
    argument uses.

    Args:
        test: "A" or "B".
        delta: concentration parameter.

    Returns:
        Scalar entropy in nats (an upper bound for Test B).
    """
    if test == "A":
        return np.log(2 * np.pi * np.e) + np.log(delta)
    return np.log(2.0) + 0.5 * np.log(2 * np.pi * np.e * delta**2)


def w2_squared(test, delta, mode=MODE):
    """Squared 2-Wasserstein distance between mu_0 and the target.

    Exact for Test A, where both are centered Gaussians with a common eigenbasis. For
    Test B this returns the note's bound E|X_0 - 2 sgn(X_0)|^2 + delta^2, attained by
    the sign-matching coupling rather than the optimal one.

    Args:
        test: "A" or "B".
        delta: concentration parameter.
        mode: Test B mode location.

    Returns:
        Scalar squared distance (an upper bound for Test B).
    """
    if test == "A":
        return (1.0 - float(delta)) ** 2
    # E|X_0 - mode sgn(X_0)|^2 = 1 + mode^2 - 2 mode E|X_0|, with E|X_0| = sqrt(2/pi).
    return 1.0 + mode**2 - 2.0 * mode * np.sqrt(2.0 / np.pi) + float(delta) ** 2


def sb_lower_bound(test, delta, eps=1.0, T=1.0):
    """Proposition 1 lower bound on the SB cost, which diverges like 2 eps log(1/delta).

    Args:
        test: "A" or "B".
        delta: concentration parameter.
        eps: diffusion scale.
        T: horizon.

    Returns:
        Scalar lower bound -2 eps H(mu_T) + eps d log(2 pi eps T) + W2^2 / T.
    """
    d = dim_of(test)
    return (-2.0 * eps * entropy(test, delta)
            + eps * d * np.log(2 * np.pi * eps * T)
            + w2_squared(test, delta) / T)


def sbb_upper_bound(test, delta, beta, eps=1.0, T=1.0, n_grid=20001):
    """Proposition 3 upper bound on the SBB cost, uniformly bounded in delta.

    Minimises 2 W2^2/T + eps d/p + beta eps d T kappa(p) over p in (0, 1/2].

    Args:
        test: "A" or "B".
        delta: concentration parameter.
        beta: volatility penalty.
        eps: diffusion scale.
        T: horizon.
        n_grid: nodes used for the 1D minimisation over p.

    Returns:
        Scalar upper bound.
    """
    d = dim_of(test)
    p = np.linspace(1e-5, 0.5, n_grid)
    kappa = 1.0 / (2 * p + 1) - 2.0 / (p + 1) + 1.0
    return float((2.0 * w2_squared(test, delta) / T
                  + eps * d / p
                  + beta * eps * d * T * kappa).min())
