"""Train LightSBB without an inverse network, averaging exact inverses of the Bass map.

The first-order map y = x - s(t, x)/beta has its own fixed point, which misses the target at
moderate beta. The undamped exact inverse is unstable, so after one first-order stage each
stage averages the current endpoint map with the exact inverse of the new Bass map.
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

PICARD_STEP = 0.5
PICARD_MAX_ITER = 500
PICARD_TOL = 1e-5
SOLVER_BATCH = 50000


def invert_bass_map(model, x, t, beta, init=None):
    """Solve y + s(t, y)/beta = x by damped fixed-point iteration.

    Args:
        model: Trained ``LightSBM``.
        x: (n, d) points in X-space.
        t: Scalar time.
        beta: Volatility penalty.
        init: (n, d) starting points, defaulting to ``x``.

    Returns:
        Tuple of the (n, d) solution and the largest residual left, over all coordinates.
    """
    solutions, worst = [], 0.0
    for start in range(0, len(x), SOLVER_BATCH):
        x_batch = x[start:start + SOLVER_BATCH]
        y = x_batch.clone() if init is None else init[start:start + SOLVER_BATCH].clone()
        t_vec = torch.full((len(y),), float(t), device=y.device)
        for _ in range(PICARD_MAX_ITER):
            residual = (y + model.get_drift(t_vec, y).detach() / beta - x_batch)
            if residual.abs().max() < PICARD_TOL:
                break
            y = y - PICARD_STEP * residual
        residual = y + model.get_drift(t_vec, y).detach() / beta - x_batch
        worst = max(worst, float(residual.abs().max()))
        solutions.append(y.detach())
    return torch.cat(solutions), worst


def update_endpoint_map(model, x, y, t, beta, stage, damping, warm_stages):
    """Apply one outer update of the endpoint map.

    Args:
        model: ``LightSBM`` trained at this stage.
        x: (n, d) points in X-space.
        y: (n, d) current images of ``x``.
        t: Scalar time of the endpoint.
        beta: Volatility penalty.
        stage: Index of the stage just trained.
        damping: Weight of the new exact inverse.
        warm_stages: Number of initial stages using the first-order map.

    Returns:
        Tuple of the (n, d) updated images and the largest solver residual.
    """
    if stage < warm_stages:
        t_vec = torch.full((len(x),), float(t), device=x.device)
        return (x - model.get_drift(t_vec, x).detach() / beta), 0.0
    exact, residual = invert_bass_map(model, x, t, beta, init=y)
    return (1.0 - damping) * y + damping * exact, residual


def encode_y0(snapshots, x_0, beta, damping, warm_stages):
    """Apply the endpoint map at t = 0 to samples the training has not seen.

    Args:
        snapshots: Models saved after each outer update, in order.
        x_0: (n, d) source samples.
        beta: Volatility penalty.
        damping: Weight of the new exact inverse.
        warm_stages: Number of initial stages using the first-order map.

    Returns:
        (n, d) points in Y-space.
    """
    y = x_0.clone()
    for stage, model in enumerate(snapshots):
        y, _ = update_endpoint_map(model, x_0, y, 0.0, beta, stage, damping, warm_stages)
    return y


def training_sbb_damped(sampler_x, sampler_y, model, beta, K, lr=1e-3, n_epochs=20000, min_epoch=20000,
                        batch_size=512, eps=1.0, safe_t=1e-2, print_every=2000, damping=0.5,
                        warm_stages=1, pool_size=100000, on_stage_end=None, device="cpu"):
    """Train LightSBB on endpoint pools refreshed by the damped exact inverse.

    The maps are stored as their values on fixed pools of samples, so each outer update solves
    the inverse once rather than once per batch.

    Args:
        sampler_x: Source sampler exposing ``sample(batch_size)``.
        sampler_y: Target sampler exposing ``sample(batch_size)``.
        model: ``LightSBM`` instance.
        beta: Volatility penalty.
        K: Number of outer stages.
        lr: Learning rate.
        n_epochs: Base epochs per stage before the decay schedule.
        min_epoch: Minimum number of epochs per stage.
        batch_size: Minibatch size.
        eps: Diffusion variance scale.
        safe_t: Margin avoiding the singularity at t = 1.
        print_every: Epochs between progress prints.
        damping: Weight of the new exact inverse.
        warm_stages: Number of initial stages using the first-order map.
        pool_size: Number of samples per endpoint.
        on_stage_end: Optional callable receiving the stage index, the model and the snapshots
            its endpoint map is made of.
        device: Training device.

    Returns:
        Tuple ``(model, snapshots, residuals)``, with the models needed by ``encode_y0`` and
        the largest solver residual of each outer update.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    x_0, x_T = sampler_x.sample(pool_size).to(device), sampler_y.sample(pool_size).to(device)
    y_0, y_T = x_0.clone(), x_T.clone()
    snapshots, residuals = [], []

    for k in range(K):
        print(f"\nTraining s^{k + 1}: ")
        curr_epoch = max(min_epoch, int(n_epochs * np.exp(-0.2 * k)))

        for epoch in range(curr_epoch):
            y_0_ = y_0[torch.randint(pool_size, (batch_size,), device=device)]
            y_T_ = y_T[torch.randint(pool_size, (batch_size,), device=device)]

            t = torch.FloatTensor(batch_size, 1).uniform_(0, 1 - safe_t).to(device)
            y_t = y_T_ * t + y_0_ * (1 - t) + torch.sqrt(eps * t * (1 - t)) * torch.randn_like(y_0_)
            loss = F.mse_loss((y_T_ - y_t) / (1 - t), model.get_drift(t.squeeze(), y_t))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if (epoch + 1) % print_every == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1}/{curr_epoch}] - Training Loss: {loss.item():.4f}")

        model.eval()
        if on_stage_end is not None:
            on_stage_end(k, model, snapshots)

        if k < K - 1:
            snapshots.append(copy.deepcopy(model))
            y_0, residual_0 = update_endpoint_map(model, x_0, y_0, 0.0, beta, k, damping, warm_stages)
            y_T, residual_T = update_endpoint_map(model, x_T, y_T, 1 - safe_t, beta, k, damping, warm_stages)
            residuals.append(max(residual_0, residual_T))
            print(f"Largest inverse residual after stage {k + 1}: {residuals[-1]:.2e}")
            if not (torch.isfinite(y_0).all() and torch.isfinite(y_T).all()):
                raise ValueError(f"non-finite endpoint map after stage {k + 1}")

    return model.eval(), snapshots, residuals
