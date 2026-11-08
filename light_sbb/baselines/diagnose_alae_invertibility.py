"""Measure curvature and inverse-map accuracy of a saved ALAE LightSBB-M checkpoint.

Run from light_sbb/, on state dicts saved with pickle as in ``run_alae.py``:

    python baselines/diagnose_alae_invertibility.py --model PATH --model-inv PATH --beta 1 --device cuda:0
"""

import argparse
import json
import pickle
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch

LIGHT_SBB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIGHT_SBB))

from baselines.alae_data import ROOT, load_alae_splits  # noqa: E402
from lightsbm import LightSBM, MLP_network  # noqa: E402

RESULTS = ROOT / "results" / "alae_invertibility"
ARCHIVE = ROOT / "alae_invertibility.tar.gz"
T_MODEL = 32
D_MODEL = 128
# Components below this prior weight are reported separately from the live ones.
LIVE_ALPHA = 1e-3


def parse_args():
    """Parse the checkpoint paths and the evaluation settings."""
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="pickled state dict of the bridge")
    p.add_argument("--model-inv", required=True, help="pickled state dict of the inverse network")
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--safe-t", type=float, default=0.01, help="margin of the Bass map at the terminal time")
    p.add_argument("--times", type=float, nargs="+", default=[0.0, 0.5, 0.9, 0.99])
    p.add_argument("--n-points", type=int, default=256, help="points per time for the Jacobian")
    p.add_argument("--n-inverse", type=int, default=2000, help="points for the inverse residuals")
    p.add_argument("--jacobian-batch", type=int, default=64)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--tag", help="results folder name; the checkpoint stem otherwise")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def load_models(args, device):
    """Rebuild both networks from their state dicts.

    Args:
        args: Parsed arguments holding the checkpoint paths.
        device: Device the networks are moved to.

    Returns:
        Tuple ``(model, model_inv)`` in evaluation mode.
    """
    with open(args.model, "rb") as f:
        state = pickle.load(f)
    with open(args.model_inv, "rb") as f:
        state_inv = pickle.load(f)

    n_potentials, dim = state["r"].shape
    # epsilon must be a float here: load_state_dict would truncate it into an integer buffer.
    model = LightSBM(dim=dim, n_potentials=n_potentials, epsilon=float(state["epsilon"]),
                     is_diagonal=True, sampling_batch_size=256)
    model.load_state_dict(state)
    model_inv = MLP_network(input_dim=dim, t_model=T_MODEL, d_model=D_MODEL)
    model_inv.load_state_dict(state_inv)
    # The Hessian cancels two terms of size 1/(1-t), which float32 cannot resolve near t = 1.
    return model.double().to(device).eval(), model_inv.to(device).eval()


def drift(model, t, y):
    """Evaluate the drift differentiably in ``y``, unlike ``LightSBM.get_drift``.

    Args:
        model: Diagonal ``LightSBM``.
        t: (n,) times in [0, 1).
        y: (n, dim) states requiring grad.

    Returns:
        (n, dim) drift, with a graph back to ``y``.
    """
    eps = model.epsilon
    r = model.get_r()
    S = torch.exp(model.S_log_diagonal_matrix)
    A = (t / (eps * (1 - t)))[:, None, None] + 1 / (eps * S)[None, :, :]
    S_log_det = model.S_log_diagonal_matrix.sum(-1)
    c = ((1 / (eps * (1 - t)))[:, None] * y)[:, None, :] + (r / (eps * S))[None, :, :]
    exp_arg = (model.log_alpha[None, :] - 0.5 * S_log_det[None, :] - 0.5 * torch.log(A).sum(-1)
               - 0.5 * ((r * r / S) / eps).sum(-1)[None, :] + 0.5 * (c * c / A).sum(-1))
    lse = torch.logsumexp(exp_arg, dim=-1)
    grad = torch.autograd.grad(lse.sum(), y, create_graph=True)[0]
    return -y / (1 - t[:, None]) + eps * grad


def bass_hessian(model, t, y, beta):
    """Return ``D²Φ_t(y) = I + D_y s(t, y) / β`` at every point of ``y``.

    Args:
        model: Diagonal ``LightSBM``.
        t: Scalar time in [0, 1).
        y: (n, dim) states.
        beta: Bass parameter.

    Returns:
        (n, dim, dim) Hessians, symmetrized, and the largest asymmetry found.
    """
    y = y.detach().requires_grad_(True)
    s = drift(model, torch.full((len(y),), t, device=y.device, dtype=y.dtype), y)
    columns = [torch.autograd.grad(s[:, i].sum(), y, retain_graph=True)[0].detach()
               for i in range(y.shape[1])]
    jac = torch.stack(columns, dim=2)
    hess = torch.eye(y.shape[1], device=y.device)[None] + jac / beta
    asymmetry = (hess - hess.transpose(1, 2)).abs().max().item()
    return 0.5 * (hess + hess.transpose(1, 2)), asymmetry


def summarize(values):
    """Return the usual statistics of a 1-d array as plain floats."""
    values = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(values, [0.01, 0.5, 0.99])
    return {"mean": float(values.mean()), "std": float(values.std()), "min": float(values.min()),
            "q01": float(quantiles[0]), "median": float(quantiles[1]), "q99": float(quantiles[2]),
            "max": float(values.max()), "n": int(len(values))}


def weight_diagnostics(model, y_0, beta):
    """Compare the mixture variances with the curvature threshold ``1/(β+1)``.

    Args:
        model: Diagonal ``LightSBM``.
        y_0: (n, dim) source points in Y-space, for the responsibilities.
        beta: Bass parameter.

    Returns:
        Dict of JSON-ready statistics.
    """
    threshold = 1.0 / (beta + 1.0)
    S = torch.exp(model.S_log_diagonal_matrix).detach()
    alpha = torch.softmax(model.log_alpha.detach(), dim=0)
    live = alpha > LIVE_ALPHA

    x_S_x = (y_0[:, None, :] * S[None] * y_0[:, None, :]).sum(-1)
    x_r = (y_0[:, None, :] * model.r[None]).sum(-1)
    logits = (x_S_x + 2 * x_r) / (2 * model.epsilon) + model.log_alpha[None]
    resp = torch.softmax(logits, dim=1).detach()
    mean_resp = resp.mean(0)
    dominant = int(mean_resp.argmax())

    return {
        "threshold": threshold,
        "alpha": alpha.tolist(),
        "n_live_alpha": int(live.sum()),
        "mean_responsibility_on_y0": mean_resp.tolist(),
        "effective_components": float(1.0 / (mean_resp ** 2).sum()),
        "S_min_all": float(S.min()),
        "S_min_live": float(S[live].min()),
        "S_median_per_component": S.median(dim=1).values.tolist(),
        "coords_below_threshold_per_component": (S < threshold).sum(dim=1).tolist(),
        "dominant_component": dominant,
        "S_dominant": summarize(S[dominant].cpu().numpy()),
        "coords_below_threshold_dominant": int((S[dominant] < threshold).sum()),
        "coords_with_S_max_below_threshold": int((S.max(dim=0).values < threshold).sum()),
        "sufficient_condition_holds": bool(S.min() > threshold),
    }


@torch.no_grad()
def inverse_residuals(model, model_inv, t, x, beta):
    """Measure how well the inverse network undoes the Bass map at time ``t``.

    Args:
        model: Diagonal ``LightSBM``.
        model_inv: Inverse network.
        t: Scalar time in [0, 1).
        x: (n, dim) points in X-space.
        beta: Bass parameter.

    Returns:
        Dict of absolute and relative residual statistics.
    """
    y = model_inv(torch.full((len(x), 1), t, device=x.device), x).double()
    x = x.double()
    with torch.enable_grad():
        s = model.get_drift(torch.full((len(x),), t, device=x.device, dtype=x.dtype), y).detach()
    error = (y + s / beta - x).norm(dim=1)
    return {"residual": summarize(error.cpu().numpy()),
            "relative_to_x": summarize((error / x.norm(dim=1)).cpu().numpy()),
            "relative_to_displacement": summarize((error / (y - x).norm(dim=1)).cpu().numpy())}


def curvature_along_path(model, y_0, y_T, args):
    """Report the spectrum of ``D²Φ_t`` on Brownian-bridge states between the endpoints.

    Args:
        model: Diagonal ``LightSBM``.
        y_0: (n, dim) Y-space sources.
        y_T: (n, dim) Y-space endpoints coupled with ``y_0``.
        args: Parsed arguments.

    Returns:
        Dict keyed by time.
    """
    out = {}
    for t in args.times:
        noise = torch.randn_like(y_0) * (t * (1 - t) * model.epsilon).sqrt()
        y_t = t * y_T + (1 - t) * y_0 + noise
        lowest, diagonal, asymmetry = [], [], 0.0
        for batch in y_t.split(args.jacobian_batch):
            hess, batch_asymmetry = bass_hessian(model, t, batch, args.beta)
            lowest.append(torch.linalg.eigvalsh(hess)[:, 0].cpu())
            diagonal.append(torch.diagonal(hess, dim1=1, dim2=2).min(dim=1).values.cpu())
            asymmetry = max(asymmetry, batch_asymmetry)
        lowest = torch.cat(lowest).numpy()
        out[f"{t:g}"] = {"min_eigenvalue": summarize(lowest),
                         "share_points_not_positive": float((lowest <= 0).mean()),
                         "min_diagonal_entry": summarize(torch.cat(diagonal).numpy()),
                         "max_asymmetry": asymmetry}
        print(f"t={t:g} min eig {lowest.min():.4f} median {np.median(lowest):.4f} "
              f"share<=0 {(lowest <= 0).mean():.3f}", flush=True)
    return out


def main():
    """Run the diagnostics on one checkpoint and archive every result so far."""
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model, model_inv = load_models(args, device)

    splits = load_alae_splits(args.data_dir)
    adults = splits["test_latents"][splits["x_inds_test"]]
    children = splits["test_latents"][splits["y_inds_test"]]
    idx = torch.randperm(len(adults))[:args.n_inverse]
    x_0 = torch.as_tensor(adults[idx.numpy()], dtype=torch.float32, device=device)
    idx = torch.randperm(len(children))[:args.n_inverse]
    x_T = torch.as_tensor(children[idx.numpy()], dtype=torch.float32, device=device)

    with torch.no_grad():
        y_0 = model_inv(torch.zeros((len(x_0), 1), device=device), x_0).double()
        y_T = model(y_0)

    # Guards the copy of the drift formula against the implementation used in training.
    probe = y_0[:32].detach().requires_grad_(True)
    t_probe = torch.full((32,), 0.5, device=device, dtype=probe.dtype)
    drift_gap = (drift(model, t_probe, probe) - model.get_drift(t_probe, probe)).abs().max().item()

    n = min(args.n_points, len(y_0))
    t_end = 1.0 - args.safe_t
    results = {
        "epsilon": float(model.epsilon),
        "beta": args.beta,
        "n_potentials": model.n_potentials,
        "drift_copy_max_gap": drift_gap,
        "weights": weight_diagnostics(model, y_0, args.beta),
        "inverse_at_t0_on_adults": inverse_residuals(model, model_inv, 0.0, x_0, args.beta),
        "inverse_at_terminal_on_children": inverse_residuals(model, model_inv, t_end, x_T, args.beta),
        "curvature": curvature_along_path(model, y_0[:n], y_T[:n], args),
    }

    out_dir = RESULTS / (args.tag or Path(args.model).stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diagnostics.json").write_text(json.dumps(results, indent=2))
    with tarfile.open(ARCHIVE, "w:gz") as tar:
        tar.add(RESULTS, arcname=RESULTS.relative_to(ROOT).as_posix())
    print(f"wrote {out_dir} and {ARCHIVE}", flush=True)


if __name__ == "__main__":
    main()
