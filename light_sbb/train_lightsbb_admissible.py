"""Train LightSBB with admissible curvature and damped exact inverses of the Bass map.

Differs from ``train_lightsbb.py`` on three points: the mixture variances are projected above
the curvature threshold ``1/(beta+1)``, the inverse network is replaced by a Newton solver,
and the endpoint maps are damped, since the undamped outer iteration is unstable at small beta.
"""

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

NEWTON_ITERS = 100
PLAIN_NEWTON_ITERS = 30
PATH_NEWTON_ITERS = 20
MAX_TEMPERATURE = 1e5
TEMPERATURE_STEPS = 40
PATH_TOLERANCE = 1e-4
LINE_SEARCH_HALVINGS = 40
ARMIJO = 1e-4
OBJECTIVE_SLACK = 1e-12
RELATIVE_TOLERANCE = 1e-9
SOLVER_BATCH = 4096
N_LOG_POINTS = 2048
AT_FLOOR_SLACK = 1e-6


def curvature_floor(beta, margin):
    """Return the smallest allowed mixture variance, or None when no floor is requested."""
    return None if margin is None else 1.0 / (beta + 1.0) + margin


def project_variances(model, floor, shared=False):
    """Project the diagonal variances: optionally tie them across components, then clamp to ``floor``."""
    log_s = model.S_log_diagonal_matrix.data
    if shared:
        log_s[:] = log_s.mean(dim=0, keepdim=True)
    if floor is not None:
        log_s.clamp_(min=math.log(floor))


def component_terms(model, y, t):
    """Return the (n, J) component log potentials, with the (n, J, d) ``c`` and (J, d) ``A`` behind them."""
    eps = model.epsilon
    S = torch.exp(model.S_log_diagonal_matrix)
    r = model.r
    A = t / (eps * (1 - t)) + 1 / (eps * S)
    c = y[:, None, :] / (eps * (1 - t)) + (r / (eps * S))[None]
    logits = (model.log_alpha - 0.5 * model.S_log_diagonal_matrix.sum(-1) - 0.5 * torch.log(A).sum(-1)
              - 0.5 * (r * r / (eps * S)).sum(-1))[None] + 0.5 * (c * c / A).sum(-1)
    return logits, c, A


def mixture_terms(model, y, t, tau=1.0):
    """Return the mixture quantities behind the drift at scalar time ``t``.

    Args:
        model: Diagonal ``LightSBM``.
        y: (n, d) states, in the dtype of the model.
        t: Scalar time in [0, 1).
        tau: Temperature of the responsibilities, 1 for the true mixture.

    Returns:
        Tuple of the (n, J) responsibilities, the (n, J, d) gradients of each component's log
        potential, and the (J, d) diagonal Hessians of those log potentials.
    """
    logits, c, A = component_terms(model, y, t)
    scale = model.epsilon * (1 - t)
    return torch.softmax(logits / tau, dim=1), c / (A * scale), 1 / (A * scale ** 2)


def convex_objective(model, y, t, beta, x, tau=1.0):
    """Return the (n,) values of ``Φ_t(y) - x·y``, whose gradient is the Bass residual."""
    log_h = tau * torch.logsumexp(component_terms(model, y, t)[0] / tau, dim=1)
    quadratic = (y * y).sum(1) * (0.5 - 0.5 / (beta * (1 - t)))
    return quadratic + model.epsilon * log_h / beta - (x * y).sum(1)


def bass_residual(model, y, t, beta, x, tau=1.0):
    """Return ``y + s(t, y)/beta - x`` from the closed-form drift, without autograd."""
    pi, grads, _ = mixture_terms(model, y, t, tau)
    drift = -y / (1 - t) + model.epsilon * (pi[..., None] * grads).sum(1)
    return y + drift / beta - x


def newton_direction(model, y, t, beta, residual, tau=1.0):
    """Solve ``D²Φ_t(y) p = residual`` exactly, as a diagonal plus a rank-J term (Woodbury).

    Args:
        model: Diagonal ``LightSBM`` with variances above the curvature floor.
        y: (n, d) states.
        t: Scalar time in [0, 1).
        beta: Bass parameter.
        residual: (n, d) right-hand side.
        tau: Temperature of the responsibilities.

    Returns:
        (n, d) Newton direction.
    """
    pi, grads, hess = mixture_terms(model, y, t, tau)
    eps = model.epsilon
    diagonal = 1 + (-1 / (1 - t) + eps * (pi[..., None] * hess[None]).sum(1)) / beta
    mean_grad = (pi[..., None] * grads).sum(1, keepdim=True)
    low_rank = torch.sqrt(eps * pi / (beta * tau))[..., None] * (grads - mean_grad)
    scaled = low_rank / diagonal[:, None, :]
    inner = torch.eye(pi.shape[1], dtype=y.dtype, device=y.device)[None] + low_rank @ scaled.transpose(1, 2)
    rhs = (low_rank @ residual[..., None])
    correction = torch.cholesky_solve(rhs, torch.linalg.cholesky(inner))
    return residual / diagonal - (scaled.transpose(1, 2) @ correction)[..., 0]


def _newton(model, x, y, t, beta, tau, iterations, tolerance):
    """Run damped Newton in place on ``y``, each point backtracking on the strictly convex objective.

    Returns:
        (n,) residual norms at the true temperature of this call.
    """
    residual = bass_residual(model, y, t, beta, x, tau)
    norm = residual.norm(dim=1)
    for _ in range(iterations):
        active = (norm > tolerance).nonzero()[:, 0]
        if len(active) == 0:
            break
        y_a, x_a, residual_a = y[active], x[active], residual[active]
        step = newton_direction(model, y_a, t, beta, residual_a, tau)
        slope = (residual_a * step).sum(1)
        value = convex_objective(model, y_a, t, beta, x_a, tau)
        # The slack absorbs rounding once the decrease is below float64 resolution.
        slack = OBJECTIVE_SLACK * (1 + value.abs())
        alpha = torch.ones(len(y_a), dtype=y.dtype, device=y.device)
        candidate = y_a
        for _ in range(LINE_SEARCH_HALVINGS):
            candidate = y_a - alpha[:, None] * step
            rejected = convex_objective(model, candidate, t, beta, x_a, tau) > value - ARMIJO * alpha * slope + slack
            if not rejected.any():
                break
            alpha[rejected] *= 0.5
        y[active] = candidate
        residual[active] = bass_residual(model, candidate, t, beta, x_a, tau)
        norm[active] = residual[active].norm(dim=1)
    return norm


def _solve_batch(model, x, y, t, beta):
    """Solve one batch, following a temperature path for the points plain Newton cannot reach.

    In high dimension the responsibilities switch almost as steps, so the potential has near-kinks
    on which Newton stalls. Raising the temperature smooths them and keeps the objective convex.
    """
    tolerance = RELATIVE_TOLERANCE * (1 + x.norm(dim=1))
    norm = _newton(model, x, y, t, beta, 1.0, PLAIN_NEWTON_ITERS, tolerance)
    failed = (norm > tolerance).nonzero()[:, 0]
    if len(failed):
        x_f, y_f, tolerance_f = x[failed], y[failed].clone(), tolerance[failed]
        for tau in torch.logspace(math.log10(MAX_TEMPERATURE), 0.0, TEMPERATURE_STEPS)[:-1].tolist():
            _newton(model, x_f, y_f, t, beta, tau, PATH_NEWTON_ITERS, PATH_TOLERANCE * (1 + x_f.norm(dim=1)))
        norm_f = _newton(model, x_f, y_f, t, beta, 1.0, NEWTON_ITERS, tolerance_f)
        better = norm_f < norm[failed]
        y[failed[better]] = y_f[better]
        norm[failed[better]] = norm_f[better]
    return y, norm


def invert_bass_map(model, x, t, beta, init=None):
    """Solve ``y + s(t, y)/beta = x`` by damped Newton in float64.

    Near ``t = 1`` the drift cancels two terms of size ``1/(1-t)``, which float32 cannot resolve.

    Args:
        model: Trained diagonal ``LightSBM``.
        x: (n, d) points in X-space.
        t: Scalar time.
        beta: Bass parameter.
        init: (n, d) starting points, ``x`` by default.

    Returns:
        Tuple of the (n, d) solution in the dtype of ``x`` and the (n,) residual norms.
    """
    precise = copy.deepcopy(model).double().eval()
    start = x if init is None else init
    solutions, residuals = [], []
    with torch.no_grad():
        for x_batch, y_batch in zip(x.double().split(SOLVER_BATCH), start.double().split(SOLVER_BATCH)):
            y, norm = _solve_batch(precise, x_batch, y_batch.clone(), t, beta)
            solutions.append(y)
            residuals.append(norm)
    return torch.cat(solutions).to(x.dtype), torch.cat(residuals).to(x.dtype)


def bass_map(model, y, t, beta):
    """Return ``y + s(t, y)/beta`` at scalar time ``t``, detached."""
    with torch.no_grad():
        return bass_residual(model, y, t, beta, torch.zeros_like(y))


def mixture_usage(model, y, floor):
    """Measure how the mixture is used on source points.

    Args:
        model: Diagonal ``LightSBM``.
        y: (n, d) source points in Y-space.
        floor: Curvature floor, or None.

    Returns:
        Dict with the effective number of components, the mean largest responsibility and, when
        there is a floor, the responsibility-weighted share of coordinates sitting on it.
    """
    precise = copy.deepcopy(model).double().eval()
    with torch.no_grad():
        pi = mixture_terms(precise, y.double(), 0.0)[0]
    mean_pi = pi.mean(0)
    usage = {"effective_components": float(1.0 / (mean_pi ** 2).sum()),
             "mean_largest_responsibility": float(pi.max(dim=1).values.mean())}
    if floor is not None:
        at_floor = (precise.S_log_diagonal_matrix <= math.log(floor) + AT_FLOOR_SLACK).double().mean(dim=1)
        usage["share_coords_at_floor"] = float((mean_pi * at_floor).sum())
    return usage


def update_endpoints(model, x, y, t, beta, stage, damping):
    """Apply one outer update of an endpoint map on its sample pool.

    Args:
        model: ``LightSBM`` trained at this stage.
        x: (n, d) points in X-space.
        y: (n, d) current images of ``x``.
        t: Scalar time of the endpoint.
        beta: Bass parameter.
        stage: Index of the stage just trained.
        damping: Weight of the new exact inverse.

    Returns:
        Tuple of the (n, d) updated images and the largest residual relative to ``|x|``.
    """
    if stage == 0:
        # The damped iteration diverges from the identity on contracting coordinates.
        return 2 * x - bass_map(model, x, t, beta), 0.0
    exact, residual = invert_bass_map(model, x, t, beta, init=y)
    return (1.0 - damping) * y + damping * exact, float((residual / x.norm(dim=1)).max())


def training_sbb_admissible(x_pool, y_pool, model, beta, K=30, lr=1e-3, n_epochs=15000, min_epoch=5000,
                            batch_size=512, eps=0.1, safe_t=1e-2, damping=0.2, margin=0.05, shared_s=False,
                            print_every=4000, device="cpu"):
    """Train LightSBB on endpoint pools refreshed by the damped exact inverse.

    With ``K = 1`` and ``margin = None`` this is plain LightSB-M bridge matching.

    Args:
        x_pool: (n, d) source samples on ``device``.
        y_pool: (m, d) target samples on ``device``.
        model: Diagonal ``LightSBM`` on ``device``.
        beta: Bass parameter.
        K: Number of outer stages.
        lr: Learning rate.
        n_epochs: Base gradient steps of a stage before the decay schedule.
        min_epoch: Minimum gradient steps of a stage.
        batch_size: Minibatch size.
        eps: Diffusion variance scale.
        safe_t: Margin avoiding the singularity at ``t = 1``.
        damping: Weight of the new exact inverse.
        margin: Distance kept above the curvature threshold, None for no floor.
        shared_s: Tie the diagonal variances across components.
        print_every: Steps between progress prints.
        device: Training device.

    Returns:
        Tuple ``(model, log)``, where ``log`` holds one dict per outer update.
    """
    floor = curvature_floor(beta, margin)
    project_variances(model, floor, shared_s)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    y_0, y_T = x_pool.clone(), y_pool.clone()
    log = []

    for k in range(K):
        print(f"\nTraining s^{k + 1}: ", flush=True)
        curr_epoch = max(min_epoch, int(n_epochs * np.exp(-0.2 * k)))
        model.train()

        for epoch in range(curr_epoch):
            y_0_ = y_0[torch.randint(len(y_0), (batch_size,), device=device)]
            y_T_ = y_T[torch.randint(len(y_T), (batch_size,), device=device)]
            t = torch.rand(batch_size, 1, device=device) * (1 - safe_t)
            y_t = y_T_ * t + y_0_ * (1 - t) + torch.sqrt(eps * t * (1 - t)) * torch.randn_like(y_0_)
            loss = F.mse_loss((y_T_ - y_t) / (1 - t), model.get_drift(t.squeeze(-1), y_t))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            project_variances(model, floor, shared_s)

            if (epoch + 1) % print_every == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1}/{curr_epoch}] - Training Loss: {loss.item():.4f}", flush=True)

        if k < K - 1:
            model.eval()
            y_0, residual_0 = update_endpoints(model, x_pool, y_0, 0.0, beta, k, damping)
            y_T, residual_T = update_endpoints(model, y_pool, y_T, 1 - safe_t, beta, k, damping)
            if not (torch.isfinite(y_0).all() and torch.isfinite(y_T).all()):
                raise ValueError(f"non-finite endpoint map after stage {k + 1}")
            entry = {"stage": k + 1, "largest_relative_residual": max(residual_0, residual_T),
                     **mixture_usage(model, y_0[:N_LOG_POINTS], floor)}
            log.append(entry)
            print(f"Stage {k + 1}: {entry}", flush=True)

    return model.eval(), log
