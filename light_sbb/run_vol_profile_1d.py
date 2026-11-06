"""Extract the learned volatility on a 1D bimodal target, where no closed form exists.

Run from light_sbb/. Results are written after every run, finished runs are skipped, and
--configs lets each run go to its own device:

    python run_vol_profile_1d.py --device cuda:0
    python run_vol_profile_1d.py --configs b10_small --device cuda:1
"""

import argparse
import csv
import json
import os
import sys
import tarfile
import traceback
from pathlib import Path

import numpy as np
import torch

LIGHT_SBB = Path(__file__).resolve().parent
sys.path.insert(0, str(LIGHT_SBB))
sys.path.insert(0, str(LIGHT_SBB / "ground_truth"))

from control_extraction import bridge_y, map_x, sigma_x  # noqa: E402
from lightsbm import LightSBM, MLP_network  # noqa: E402
from train_lightsbb import training_sbb  # noqa: E402
from train_lightsbb_beta_large import training_sbb_beta_large  # noqa: E402

ROOT = LIGHT_SBB.parent
RESULTS_DIR = ROOT / "results" / "vol_profile_1d"
ARCHIVE = ROOT / "vol_profile_1d.tar.gz"

# name: (beta, training loop). "large" needs no inverse network, "small" learns one, and
# "sb" is the Schrodinger bridge reference, whose loop at K = 1 never reads beta.
CONFIGS = {
    "b100_large": (100.0, "large"),
    "b10_large": (10.0, "large"),
    "b10_small": (10.0, "small"),
    "sb": (None, "sb"),
}
MODE = 2.0
SAMPLING_BATCH = 4096
PROFILE_TIMES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.99)
RESIDUAL_TIMES = (0.0, 0.3, 0.6, 0.99)
N_GRID = 201
N_PATH = 5000
SUMMARY_FIELDS = ["config", "beta", "loop", "std_neg", "std_pos", "target_std", "mode_err",
                  "mass_right", "s_max", "s_threshold", "frac_sigma_negative_on_path",
                  "sigma_min_on_path", "sigma_max_on_path", "sigma_at_mode_t99",
                  "sigma_at_center_t99", "inverse_residual_max"]


class StandardNormal:
    """Sample from the 1D source N(0, 1)."""

    def __init__(self, device):
        self.device = device

    def sample(self, n):
        """Return (n, 1) samples."""
        return torch.randn(n, 1, device=self.device)


class Bimodal:
    """Sample from the 1D target 0.5 N(-MODE, delta^2) + 0.5 N(+MODE, delta^2)."""

    def __init__(self, delta, device):
        self.delta = delta
        self.device = device

    def sample(self, n):
        """Return (n, 1) samples."""
        signs = torch.randint(0, 2, (n, 1), device=self.device, dtype=torch.float32) * 2.0 - 1.0
        return signs * MODE + self.delta * torch.randn(n, 1, device=self.device)


def parse_args():
    """Parse the runs to launch, the device and the training settings."""
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", choices=sorted(CONFIGS), default=list(CONFIGS))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--delta", type=float, default=0.02, help="standard deviation of each mode")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eps", type=float, default=1.0)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n-epochs", type=int, default=20000)
    p.add_argument("--min-epoch", type=int, default=20000)
    p.add_argument("--n-potentials", type=int, default=50)
    p.add_argument("--s-init", type=float, default=0.1)
    p.add_argument("--t-model", type=int, default=8)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--safe-t", type=float, default=1e-2)
    p.add_argument("--print-every", type=int, default=5000)
    p.add_argument("--n-eval", type=int, default=50000)
    return p.parse_args()


def write_atomic(path, write):
    """Write through a temporary file, so concurrent runs never read a partial file.

    Args:
        path: Destination path.
        write: Callable receiving the temporary path and creating the file there.
    """
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    write(tmp)
    os.replace(tmp, path)


def train(beta, loop, source, target, args, device):
    """Train one configuration.

    Args:
        beta: Volatility penalty, None for the Schrodinger bridge.
        loop: One of "large", "small" or "sb".
        source: Source sampler.
        target: Target sampler.
        args: Parsed CLI arguments.
        device: Training device.

    Returns:
        Tuple ``(model, model_inv)``, with ``model_inv`` None unless the loop learns one.
    """
    model = LightSBM(dim=1, n_potentials=args.n_potentials, epsilon=args.eps,
                     S_diagonal_init=args.s_init, is_diagonal=True,
                     sampling_batch_size=SAMPLING_BATCH).to(device)
    init = torch.cat([source.sample(args.n_potentials // 2),
                      target.sample(args.n_potentials - args.n_potentials // 2)])
    model.init_r_by_samples(init)

    shared = dict(n_epochs=args.n_epochs, min_epoch=args.min_epoch, batch_size=args.batch_size,
                  lr=args.lr, eps=args.eps, safe_t=args.safe_t, print_every=args.print_every,
                  device=device)
    if loop == "sb":
        return training_sbb_beta_large(source, target, model, beta=1.0, K=1, **shared), None
    if loop == "large":
        return training_sbb_beta_large(source, target, model, beta, K=args.k, **shared), None

    model_inv = MLP_network(input_dim=1, t_model=args.t_model, d_model=args.d_model).to(device)
    return training_sbb(source, target, model, model_inv, beta, K=args.k, **shared)


def endpoints(model, model_inv, beta, loop, source, args, device):
    """Sample the learned coupling.

    Args:
        model: Trained ``LightSBM``.
        model_inv: Inverse network, or None.
        beta: Volatility penalty, None for the Schrodinger bridge.
        loop: One of "large", "small" or "sb".
        source: Source sampler.
        args: Parsed CLI arguments.
        device: Evaluation device.

    Returns:
        Tuple of the (n, 1) tensors ``(y_0, y_T, x_T)``.
    """
    x_0 = source.sample(args.n_eval)
    n = len(x_0)
    if loop == "sb":
        y_0 = x_0
    elif loop == "large":
        y_0 = (x_0 - model.get_drift(torch.zeros(n, device=device), x_0) / beta).detach()
    else:
        with torch.no_grad():
            y_0 = model_inv(torch.zeros((n, 1), device=device), x_0)

    y_T = model(y_0)
    if loop == "sb":
        return y_0, y_T, y_T
    t_end = torch.full((n,), 1.0 - args.safe_t, device=device)
    return y_0, y_T, (y_T + model.get_drift(t_end, y_T) / beta).detach()


def mode_statistics(x):
    """Return the occupancy and spread of each mode.

    Args:
        x: (n,) terminal samples.

    Returns:
        Dict of summary statistics; an empty mode is charged the full distance MODE.
    """
    right = x >= 0.0
    stats = {"mass_right": float(right.mean())}
    errors = []
    for name, part in (("neg", x[~right]), ("pos", x[right])):
        stats[f"std_{name}"] = float(part.std()) if len(part) else float("nan")
        errors.append(abs(abs(float(part.mean())) - MODE) if len(part) else MODE)
    stats["mode_err"] = max(errors)
    return stats


def volatility_profile(model, beta, y_0, y_T, args):
    """Evaluate the volatility on a grid and along the learned bridge.

    Args:
        model: Trained ``LightSBM``.
        beta: Volatility penalty.
        y_0: (n, 1) bridge starts.
        y_T: (n, 1) bridge ends.
        args: Parsed CLI arguments.

    Returns:
        Tuple of the arrays to save and the scalar diagnostics.
    """
    eps = torch.as_tensor(args.eps, dtype=y_0.dtype, device=y_0.device)
    grids, xs, sigmas, path_sigmas = [], [], [], []

    for t in PROFILE_TIMES:
        y_t = bridge_y(y_0[:N_PATH], y_T[:N_PATH], t, eps)
        t_path = torch.full((len(y_t),), t, dtype=y_t.dtype, device=y_t.device)
        path_sigmas.append(sigma_x(model, t_path, y_t, beta).detach().reshape(-1).cpu().numpy())

        lo, hi = float(y_t.min()), float(y_t.max())
        grid = torch.linspace(lo, hi, N_GRID, device=y_t.device).reshape(-1, 1)
        t_grid = torch.full((N_GRID,), t, dtype=grid.dtype, device=grid.device)
        grids.append(grid.reshape(-1).cpu().numpy())
        # sigma is evaluated at Y and indexed on X, the image of the grid under the Bass map.
        xs.append(map_x(model, t_grid, grid, beta).detach().reshape(-1).cpu().numpy())
        sigmas.append(sigma_x(model, t_grid, grid, beta).detach().reshape(-1).cpu().numpy())

    path_sigmas = np.stack(path_sigmas)
    x_last, sigma_last = xs[-1], sigmas[-1]
    diagnostics = {
        "frac_sigma_negative_on_path": float((path_sigmas < 0).mean()),
        "frac_sigma_negative_by_time": (path_sigmas < 0).mean(axis=1).tolist(),
        "sigma_min_on_path": float(path_sigmas.min()),
        "sigma_max_on_path": float(path_sigmas.max()),
        "sigma_at_mode_t99": float(sigma_last[np.abs(np.abs(x_last) - MODE).argmin()]),
        "sigma_at_center_t99": float(sigma_last[np.abs(x_last).argmin()]),
    }
    arrays = {"times": np.array(PROFILE_TIMES), "grid_y": np.stack(grids), "grid_x": np.stack(xs),
              "sigma": np.stack(sigmas), "sigma_on_path": path_sigmas}
    return arrays, diagnostics


def inverse_residual(model, model_inv, beta, y_T, device):
    """Measure the round trip of the inverse network on the Bass map.

    Args:
        model: Trained ``LightSBM``.
        model_inv: Trained inverse network.
        beta: Volatility penalty.
        y_T: (n, 1) points in Y-space.
        device: Evaluation device.

    Returns:
        Dict mapping each time to the mean absolute error of ``Z(t, X_t(y))`` against ``y``.
    """
    y = y_T[:N_PATH]
    residuals = {}
    for t in RESIDUAL_TIMES:
        t_vec = torch.full((len(y),), t, device=device)
        x = map_x(model, t_vec, y, beta).detach()
        with torch.no_grad():
            recovered = model_inv(t_vec[:, None], x)
        residuals[f"{t:g}"] = float((recovered - y).abs().mean())
    return residuals


def run_one(name, args, device):
    """Train and score one configuration, then refresh the summary and the archive.

    A failed run is recorded and the others go on.

    Args:
        name: Key of ``CONFIGS``.
        args: Parsed CLI arguments.
        device: Device of this run.
    """
    beta, loop = CONFIGS[name]
    run_dir = RESULTS_DIR / f"delta_{args.delta:g}" / name
    if (run_dir / "metrics.json").exists():
        print(f"{name}: already complete, skipping", flush=True)
        return
    print(f"{name}: starting on {device}", flush=True)

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    source, target = StandardNormal(device), Bimodal(args.delta, device)

    try:
        model, model_inv = train(beta, loop, source, target, args, device)
        weights = {"model": model.state_dict(), "beta": beta, "loop": loop, "seed": args.seed,
                   "delta": args.delta, "eps": args.eps, "k": args.k,
                   "model_inv": None if model_inv is None else model_inv.state_dict()}
        write_atomic(run_dir / "weights.pt", lambda tmp: torch.save(weights, tmp))

        y_0, y_T, x_T = endpoints(model, model_inv, beta, loop, source, args, device)
        x_np = x_T.detach().reshape(-1).cpu().numpy()
        if not np.isfinite(x_np).all():
            raise ValueError("non-finite terminal samples")

        s_max = float(model.S_log_diagonal_matrix.exp().max())
        metrics = {"config": name, "beta": beta, "loop": loop, "seed": args.seed,
                   "delta": args.delta, "eps": args.eps, "k": args.k, "target_std": args.delta,
                   "s_max": s_max, **mode_statistics(x_np)}
        np.save(run_dir / "x_T.npy", x_np)

        if loop != "sb":
            # T_beta^+[phi] is -infinity below this value, at T = eps = 1 (Proposition A.1).
            metrics["s_threshold"] = 1.0 / (beta + 1.0)
            arrays, diagnostics = volatility_profile(model, beta, y_0, y_T, args)
            metrics.update(diagnostics)
            np.savez(run_dir / "profile.npz", **arrays)
        if model_inv is not None:
            residuals = inverse_residual(model, model_inv, beta, y_T, device)
            metrics["inverse_residual_by_time"] = residuals
            metrics["inverse_residual_max"] = max(residuals.values())

        write_atomic(run_dir / "metrics.json",
                     lambda tmp: Path(tmp).write_text(json.dumps(metrics, indent=2)))
        (run_dir / "failure.json").unlink(missing_ok=True)
    except (RuntimeError, ValueError) as error:
        print(f"{name}: failed with {error}", flush=True)
        (run_dir / "failure.json").write_text(json.dumps(
            {"config": name, "error": str(error), "traceback": traceback.format_exc()}, indent=2))

    write_summary()
    write_archive()


def write_summary():
    """Rebuild the summary table from every finished run."""
    rows = [json.loads(path.read_text()) for path in sorted(RESULTS_DIR.glob("*/*/metrics.json"))]

    def write(tmp):
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, ["delta"] + SUMMARY_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    write_atomic(RESULTS_DIR / "summary.csv", write)


def write_archive():
    """Bundle the results so far into one archive at the repo root."""

    def write(tmp):
        with tarfile.open(tmp, "w:gz") as tar:
            tar.add(RESULTS_DIR, arcname=RESULTS_DIR.relative_to(ROOT).as_posix())

    write_atomic(ARCHIVE, write)


def main():
    """Run each requested configuration in turn on the chosen device."""
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    for name in args.configs:
        run_one(name, args, device)
    print(f"archive: {ARCHIVE}", flush=True)


if __name__ == "__main__":
    main()
