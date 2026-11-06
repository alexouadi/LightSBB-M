"""Compute the SBB primal-dual gap of trained 2D models, from their saved weights.

The learned mixture gives the dual potential exactly, so the reduced dual functional only needs
the two infima of the Moreau transform, solved on a grid and then refined. Costs carry the factors
1/2 and beta/2 of the problem. Run from light_sbb/, after run_matched_budget_2d.py:

    python primal_dual_gap.py --device cpu
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

LIGHT_SBB = Path(__file__).resolve().parent
sys.path.insert(0, str(LIGHT_SBB))
sys.path.insert(0, str(LIGHT_SBB / "ground_truth"))

from lightsbm import LightSBM  # noqa: E402
from plan_metrics import objective  # noqa: E402

ROOT = LIGHT_SBB.parent
RESULTS_DIR = ROOT / "results" / "matched_budget_2d"
DIM = 2
N_POTENTIALS = 50
SAFE_T = 1e-2
GRID_POINTS = 200
GRID_MARGIN = 1.0
CHUNK = 256
REFINE_STEPS = 200
REFINE_LR = 0.01
N_TIMES = 21
TAIL_WIDTH = 0.005


def log_phi(model, x):
    """Evaluate the dual potential at T = 1, from the Gaussian-mixture parametrization.

    Args:
        model: Trained ``LightSBM`` in float64.
        x: (n, d) points.

    Returns:
        (n,) values of ``|x|^2/(2 eps) + log sum_j alpha_j N(x | r_j, eps S_j)``.
    """
    eps = model.epsilon
    s = torch.exp(model.S_log_diagonal_matrix)
    diff = x[:, None, :] - model.get_r()[None, :, :]
    log_normal = (-0.5 * (diff**2 / (eps * s)).sum(-1)
                  - 0.5 * torch.log(2 * math.pi * eps * s).sum(-1))
    return (x**2).sum(-1) / (2 * eps) + torch.logsumexp(model.log_alpha[None, :] + log_normal, dim=-1)


def log_h0(model, y):
    """Evaluate log h_0 = log(N_{eps T} * exp(phi)) at T = 1, in closed form.

    Args:
        model: Trained ``LightSBM`` in float64.
        y: (n, d) points.

    Returns:
        (n,) values of ``log h_0(y)``.
    """
    eps = model.epsilon
    d = y.shape[1]
    s = torch.exp(model.S_log_diagonal_matrix)
    r = model.get_r()
    precision = 1 / (eps * s)
    linear = y[:, None, :] / eps + (r * precision)[None, :, :]
    log_terms = (model.log_alpha[None, :]
                 - 0.5 * torch.log(2 * math.pi * eps * s).sum(-1)[None, :]
                 - 0.5 * (r**2 * precision).sum(-1)[None, :]
                 + 0.5 * d * math.log(2 * math.pi)
                 - 0.5 * torch.log(precision).sum(-1)[None, :]
                 + 0.5 * (linear**2 / precision[None, :, :]).sum(-1))
    log_kernel = -(y**2).sum(-1) / (2 * eps) - 0.5 * d * torch.log(2 * math.pi * eps)
    return log_kernel + torch.logsumexp(log_terms, dim=-1)


def moreau(potential, x, beta, grid_points=GRID_POINTS):
    """Compute inf_y {potential(y) + (beta/2) |x - y|^2} on a 2D grid, then refine it.

    Args:
        potential: Callable mapping (n, 2) points to (n,) values.
        x: (n, 2) points where the transform is evaluated.
        beta: Volatility penalty.
        grid_points: Grid points per axis.

    Returns:
        (n,) values of the transform.
    """
    low, high = x.min(0).values - GRID_MARGIN, x.max(0).values + GRID_MARGIN
    axes = [torch.linspace(low[i], high[i], grid_points, dtype=x.dtype, device=x.device) for i in range(2)]
    grid = torch.cartesian_prod(*axes)
    with torch.no_grad():
        values = torch.cat([potential(batch) for batch in grid.split(CHUNK * 64)])
        best_value, best_point = [], []
        for batch in x.split(CHUNK):
            total = values[None, :] + 0.5 * beta * torch.cdist(batch, grid) ** 2
            value, index = total.min(dim=1)
            best_value.append(value)
            best_point.append(grid[index])
    best_value, y = torch.cat(best_value), torch.cat(best_point).clone().requires_grad_(True)

    optimizer = torch.optim.Adam([y], lr=REFINE_LR)
    for _ in range(REFINE_STEPS):
        optimizer.zero_grad()
        (potential(y) + 0.5 * beta * ((x - y) ** 2).sum(-1)).sum().backward()
        optimizer.step()
    with torch.no_grad():
        refined = potential(y) + 0.5 * beta * ((x - y) ** 2).sum(-1)
    return torch.minimum(best_value, refined)


def reduced_dual(model, x_0, x_T, beta):
    """Evaluate the reduced dual functional at the learned potential.

    With the costs of the problem at scale eps, the potentials enter multiplied by eps.

    Args:
        model: Trained ``LightSBM`` in float64.
        x_0: (n, 2) source samples.
        x_T: (n, 2) terminal samples.
        beta: Volatility penalty.

    Returns:
        Tuple of the value of the dual functional and its Monte Carlo standard error.
    """
    eps = model.epsilon
    terminal = moreau(lambda y: eps * log_phi(model, y), x_T, beta)
    initial = moreau(lambda y: eps * log_h0(model, y), x_0, beta)
    error = (terminal.var() / len(terminal) + initial.var() / len(initial)).sqrt()
    return float(terminal.mean() - initial.mean()), float(error)


def primal_cost(model, y_0, y_T, beta):
    """Estimate the primal cost on [0, T], with the factors 1/2 and beta/2.

    The controls are only evaluated up to T - SAFE_T, so the last interval is priced at the
    integrand reached there. Leaving it out would put the primal below the dual artificially.

    Args:
        model: Trained ``LightSBM`` in float64.
        y_0: (n, 2) learned bridge starts.
        y_T: (n, 2) learned bridge ends.
        beta: Volatility penalty.

    Returns:
        Estimated cost.
    """
    eps = float(model.epsilon)
    # plan_metrics.objective carries no factor 1/2 on either term.
    body = objective(model, y_0, y_T, beta, np.linspace(0.0, 1.0, N_TIMES), eps)
    end = 1.0 - SAFE_T
    rate = objective(model, y_0, y_T, beta, np.array([end - TAIL_WIDTH, end]), eps) / TAIL_WIDTH
    return 0.5 * (body + SAFE_T * rate)


def tail_margin(model, beta):
    """Return the smallest S_max over coordinates minus the threshold below which T+ is -infinity."""
    return float(torch.exp(model.S_log_diagonal_matrix).max(dim=0).values.min()) - 1.0 / (beta + 1.0)


def primal_dual(model, x_0, y_0, y_T, x_T_model, x_T_target, beta):
    """Compute the primal cost, the dual values and the gaps between them.

    Weak duality only binds the primal against the dual at the model's own terminal law. The
    value at the target is reported too, and the two differ by the marginal error.

    Args:
        model: Trained ``LightSBM`` in float64.
        x_0: (n, 2) source samples used for the dual.
        y_0: (n, 2) learned bridge starts.
        y_T: (n, 2) learned bridge ends.
        x_T_model: (n, 2) terminal samples of the model.
        x_T_target: (n, 2) target samples.
        beta: Volatility penalty.

    Returns:
        Dict with the primal, both dual values with their standard errors, both gaps relative
        to the primal, and ``tail_margin``.
    """
    primal = primal_cost(model, y_0, y_T, beta)
    dual_model, error_model = reduced_dual(model, x_0, x_T_model, beta)
    dual_target, error_target = reduced_dual(model, x_0, x_T_target, beta)
    return {"primal": primal, "dual_at_model_law": dual_model, "dual_at_model_law_se": error_model,
            "gap_at_model_law": (primal - dual_model) / abs(primal),
            "dual_at_target": dual_target, "dual_at_target_se": error_target,
            "gap_at_target": (primal - dual_target) / abs(primal),
            "tail_margin": tail_margin(model, beta)}


def load_model(path, eps, device):
    """Load saved weights into a float64 ``LightSBM``.

    Args:
        path: File holding a state dict, or a dict with the state dict under ``model``.
        eps: Diffusion scale.
        device: Evaluation device.

    Returns:
        The model in evaluation mode.
    """
    state = torch.load(path, map_location=device, weights_only=False)
    state = state.get("model", state)
    model = LightSBM(dim=DIM, n_potentials=state["log_alpha"].shape[0], epsilon=eps, is_diagonal=True,
                     sampling_batch_size=4096)
    model.load_state_dict(state)
    return model.double().to(device).eval()


def main():
    """Score every finished SBB run of the matched-budget experiment."""
    from run_matched_budget_2d import TASKS, terminal_samples
    from utils import GeneratorTwoD, wasserstein

    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-samples", type=int, default=10000)
    args = p.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    rows = []
    for path in sorted(RESULTS_DIR.glob("*/sbb/seed_*/weights.pt")):
        task, seed = path.parts[-4], int(path.parts[-2].split("_")[1])
        source_name, target_name, eps, beta, _ = TASKS[task]
        model = load_model(path, eps, device)
        torch.manual_seed(seed)
        np.random.seed(seed)
        x_0 = GeneratorTwoD(source_name, DIM, device).sample(args.n_samples).to(device).double()
        x_target = GeneratorTwoD(target_name, DIM, device).sample(args.n_samples).to(device).double()
        y_0, y_T, x_T = terminal_samples(model, "sbb", beta, x_0)
        x_T = x_T.detach()
        row = {"task": task, "seed": seed, "w2_to_target": float(wasserstein(x_T, x_target)),
               **primal_dual(model, x_0, y_0, y_T, x_T, x_target, beta)}
        rows.append(row)
        print(row, flush=True)

        with open(RESULTS_DIR / "primal_dual.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (RESULTS_DIR / "primal_dual.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
