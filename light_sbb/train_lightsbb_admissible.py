"""Train LightSBB with admissible curvature and damped exact inverses of the Bass map.

Differs from ``train_lightsbb.py`` on three points: the mixture variances are projected above
the curvature threshold ``1/(beta+1)``, the inverse network is replaced by a numerical solver,
and the endpoint maps are damped, since the undamped outer iteration is unstable at small beta.
"""

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

PICARD_STEP = 0.5
PICARD_ITERS = 300
SOLVER_BATCH = 20000


def curvature_floor(beta, margin):
    """Return the smallest allowed mixture variance for a given ``beta``."""
    return 1.0 / (beta + 1.0) + margin


def project_variances(model, floor):
    """Clamp every diagonal mixture variance to at least ``floor``."""
    model.S_log_diagonal_matrix.data.clamp_(min=math.log(floor))


def bass_map(model, y, t, beta):
    """Return ``y + s(t, y)/beta`` at scalar time ``t``, detached."""
    t_vec = torch.full((len(y),), float(t), device=y.device, dtype=y.dtype)
    return (y + model.get_drift(t_vec, y) / beta).detach()


def invert_bass_map(model, x, t, beta, init=None):
    """Solve ``y + s(t, y)/beta = x`` by damped fixed-point iteration, in float64.

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
    for x_batch, y_batch in zip(x.double().split(SOLVER_BATCH), start.double().split(SOLVER_BATCH)):
        y = y_batch.clone()
        for _ in range(PICARD_ITERS):
            y = y - PICARD_STEP * (bass_map(precise, y, t, beta) - x_batch)
        residuals.append((bass_map(precise, y, t, beta) - x_batch).norm(dim=1))
        solutions.append(y)
    return torch.cat(solutions).to(x.dtype), torch.cat(residuals).to(x.dtype)


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
        Tuple of the (n, d) updated images and the largest residual norm.
    """
    if stage == 0:
        # The damped iteration diverges from the identity on contracting coordinates.
        return 2 * x - bass_map(model, x, t, beta), 0.0
    exact, residual = invert_bass_map(model, x, t, beta, init=y)
    return (1.0 - damping) * y + damping * exact, float(residual.max())


def training_sbb_admissible(x_pool, y_pool, model, beta, K=30, lr=1e-3, n_epochs=15000, min_epoch=5000,
                            batch_size=512, eps=0.1, safe_t=1e-2, damping=0.2, margin=0.05,
                            print_every=4000, device="cpu"):
    """Train LightSBB on endpoint pools refreshed by the damped exact inverse.

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
        margin: Distance kept above the curvature threshold.
        print_every: Steps between progress prints.
        device: Training device.

    Returns:
        Tuple ``(model, log)``, where ``log`` holds the largest solver residual of each stage.
    """
    floor = curvature_floor(beta, margin)
    project_variances(model, floor)
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
            project_variances(model, floor)

            if (epoch + 1) % print_every == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1}/{curr_epoch}] - Training Loss: {loss.item():.4f}", flush=True)

        if k < K - 1:
            model.eval()
            y_0, residual_0 = update_endpoints(model, x_pool, y_0, 0.0, beta, k, damping)
            y_T, residual_T = update_endpoints(model, y_pool, y_T, 1 - safe_t, beta, k, damping)
            log.append(max(residual_0, residual_T))
            print(f"Largest inverse residual after stage {k + 1}: {log[-1]:.2e}", flush=True)
            if not (torch.isfinite(y_0).all() and torch.isfinite(y_T).all()):
                raise ValueError(f"non-finite endpoint map after stage {k + 1}")

    return model.eval(), log
