"""Ground-truth validation of the SBB plan on the analytic 2D Gaussian benchmark.

Trains LightSBB on N(0, I2) -> N(0, diag(10, 0.1)) for each beta, then reports metrics
A-D against the exact solution: plan SW2, cross-covariance error, objective gap, and
drift/volatility recovery. 

Run from inside light_sbb/ground_truth/:  python run_ground_truth.py
Results, weights and (X_0, X_1) pairs are archived to a single .tar.gz at the repo root.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import baseline_lightsb as bl  # noqa: E402
import gaussian_ground_truth as gt  # noqa: E402
import plan_metrics as pm  # noqa: E402
from control_extraction import encode_y  # noqa: E402
from lightsbm import LightSBM, MLP_network  # noqa: E402
from train_lightsbb import training_sbb  # noqa: E402
from train_lightsbb_beta_large import training_sbb_beta_large  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS_SUBDIR = "results/ground_truth"
LARGE_BETA = 100.0
SAFE_T = 1e-2


def run_dir(beta, seed, K, args=None):
    """Return the folder holding one run's outputs, creating it if needed.

    Runs that depart from the default training settings get those appended to the leaf
    name, so a sweep over safe_t or the net widths cannot overwrite the default run.

    Args:
        beta: SBB volatility penalty, or None for the beta-free baseline.
        seed: run seed.
        K: number of outer iterations.
        args: parsed CLI arguments, or None to name the folder by (beta, seed, K) alone.

    Returns:
        Path to results/ground_truth/<beta_x|lightsb>/seed_<s>/K_<k>[_tag]/.
    """
    family = "lightsb" if beta is None else f"beta_{beta:g}"
    leaf = f"K_{K}"
    if args is not None:
        for name, value, default in (("st", args.safe_t, SAFE_T),
                                     ("si", args.s_init, 0.1),
                                     ("tm", args.t_model, 8),
                                     ("dm", args.d_model, 32)):
            if value != default:
                leaf += f"_{name}{value:g}"
    path = ROOT / RESULTS_SUBDIR / family / f"seed_{seed}" / leaf
    path.mkdir(parents=True, exist_ok=True)
    return path


class GaussianSampler:
    """Sample from a centered Gaussian with diagonal covariance."""

    def __init__(self, variances, device="cpu"):
        self.std = torch.tensor(variances, dtype=torch.float32, device=device).sqrt()
        self.device = device

    def sample(self, n):
        """Return (n, d) samples on the sampler's device."""
        return torch.randn(n, len(self.std), device=self.device) * self.std


def train(x_sampler, y_sampler, beta, args, device):
    """Train one model, returning (model, model_inv) with model_inv None for large beta."""
    model = LightSBM(dim=2, n_potentials=args.n_potentials, epsilon=args.eps,
                     S_diagonal_init=args.s_init, is_diagonal=True).to(device)

    init_x = x_sampler.sample(args.n_potentials // 2)
    init_y = y_sampler.sample(args.n_potentials - args.n_potentials // 2)
    model.init_r_by_samples(torch.cat([init_x, init_y], dim=0))

    shared = dict(K=args.K, n_epochs=args.n_epochs, min_epoch=args.min_epoch,
                  batch_size=args.batch_size, lr=args.lr, eps=args.eps,
                  safe_t=args.safe_t, print_every=args.print_every, device=device)

    if beta >= LARGE_BETA:
        return training_sbb_beta_large(x_sampler, y_sampler, model, beta, **shared), None

    model_inv = MLP_network(input_dim=2, t_model=args.t_model, d_model=args.d_model).to(device)
    return training_sbb(x_sampler, y_sampler, model, model_inv, beta, **shared)


def generate_pairs(model, model_inv, x_sampler, beta, n, device, safe_t=SAFE_T):
    """Sample (X_0, X_hat_1) pairs and the Y-space endpoints the metrics need.

    Args:
        model: trained LightSBM.
        model_inv: inverse net, or None in the large-beta regime.
        x_sampler: source sampler.
        beta: SBB volatility penalty.
        n: number of pairs.
        device: torch device.
        safe_t: margin keeping the terminal time away from the singularity at t = 1.

    Returns:
        (pairs, y_0, y_T) with pairs of shape (n, 4).
    """
    x_0 = x_sampler.sample(n)
    y_0 = encode_y(model, model_inv, x_0, 0.0, beta, safe_t=safe_t)
    y_T = model(y_0)

    t_T = torch.full((n,), 1.0 - safe_t, device=device)
    x_1 = (y_T + model.get_drift(t_T, y_T) / beta).detach()
    return torch.cat([x_0, x_1], dim=1).cpu().numpy(), y_0, y_T


def train_baseline(x_sampler, y_sampler, args, device):
    """Train the LightSB-M baseline: the same loop at K = 1, where beta never enters."""
    model = LightSBM(dim=2, n_potentials=args.n_potentials, epsilon=args.eps,
                     S_diagonal_init=args.s_init, is_diagonal=True).to(device)

    init_x = x_sampler.sample(args.n_potentials // 2)
    init_y = y_sampler.sample(args.n_potentials - args.n_potentials // 2)
    model.init_r_by_samples(torch.cat([init_x, init_y], dim=0))

    return training_sbb_beta_large(
        x_sampler, y_sampler, model, beta=1.0, K=1, n_epochs=args.n_epochs,
        min_epoch=args.min_epoch, batch_size=args.batch_size, lr=args.lr, eps=args.eps,
        safe_t=args.safe_t, print_every=args.print_every, device=device)


def generate_pairs_baseline(model, x_sampler, n):
    """Sample (X_0, X_hat_1) for the baseline, where the Bass map is the identity.

    Args:
        model: trained LightSBM.
        x_sampler: source sampler.
        n: number of pairs.

    Returns:
        (pairs, y_0, y_T) with pairs of shape (n, 4).
    """
    y_0 = x_sampler.sample(n)
    y_T = model(y_0)
    return torch.cat([y_0, y_T], dim=1).detach().cpu().numpy(), y_0, y_T


def run_baseline(betas, seed, args, device):
    """Train LightSB-M once and score it against every beta column.

    The model has no beta, so one training serves all columns; only the reference it
    is compared against changes.

    Args:
        betas: sequence of beta values to score against.
        seed: run seed.
        args: parsed CLI arguments.
        device: torch device.

    Returns:
        List of metric rows, one per beta.
    """
    print(f"\n{'=' * 70}\nLightSB-M baseline (K = 1)   seed = {seed}\n{'=' * 70}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_sampler = GaussianSampler([1.0, 1.0], device)
    y_sampler = GaussianSampler(args.sigma2, device)

    started = time.time()
    model = train_baseline(x_sampler, y_sampler, args, device)
    train_time = time.time() - started

    model.eval()
    pairs, y_0, y_T = generate_pairs_baseline(model, x_sampler, args.n_eval)

    out = run_dir(None, seed, 1, args)
    np.save(out / "pairs.npy", pairs)
    torch.save({"model": model.state_dict(), "seed": seed, "eps": args.eps,
                "n_potentials": args.n_potentials}, out / "weights.pt")

    rows = []
    for beta in betas:
        metrics = bl.evaluate(model, pairs, y_0, y_T, beta, n_times=args.n_times,
                              sigma2=tuple(args.sigma2), eps=args.eps, seed=seed)
        metrics.update(beta=beta, seed=seed, K=1, method="lightsb",
                       train_time_s=round(train_time, 1))
        print(f"\n  beta = {beta:g}")
        print(f"    plan SW2 {metrics['plan_sw2']:.4f}   cross-cov {metrics['cross_cov_err']:.4f}"
              f"   obj gap {metrics['objective_gap']:+.4f}"
              f"   E_a {metrics['E_a']:.4f}   E_b {metrics['E_b']:.4f}")
        rows.append(metrics)

    with open(out / "metrics.json", "w") as f:
        json.dump(rows, f, indent=2)
    return rows


def run_one(beta, seed, args, device):
    """Train and evaluate a single (beta, seed), returning its metric row."""
    print(f"\n{'=' * 70}\nbeta = {beta:g}   seed = {seed}   K = {args.K}\n{'=' * 70}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_sampler = GaussianSampler([1.0, 1.0], device)
    y_sampler = GaussianSampler(args.sigma2, device)

    started = time.time()
    model, model_inv = train(x_sampler, y_sampler, beta, args, device)
    train_time = time.time() - started

    model.eval()
    pairs, y_0, y_T = generate_pairs(model, model_inv, x_sampler, beta, args.n_eval,
                                     device, args.safe_t)

    metrics = pm.evaluate(model, pairs, y_0, y_T, beta, n_times=args.n_times,
                          sigma2=tuple(args.sigma2), eps=args.eps, seed=seed)
    metrics.update(beta=beta, seed=seed, K=args.K, method="lightsbb",
                   safe_t=args.safe_t, s_init=args.s_init, t_model=args.t_model,
                   d_model=args.d_model, train_time_s=round(train_time, 1))

    out = run_dir(beta, seed, args.K, args)
    np.save(out / "pairs.npy", pairs)
    torch.save({"model": model.state_dict(),
                "model_inv": None if model_inv is None else model_inv.state_dict(),
                "beta": beta, "K": args.K, "seed": seed, "eps": args.eps,
                "n_potentials": args.n_potentials},
               out / "weights.pt")
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  plan SW2        {metrics['plan_sw2']:.4f}")
    print(f"  terminal SW2    {metrics['terminal_sw2']:.4f}")
    print(f"  cross-cov err   {metrics['cross_cov_err']:.4f}")
    print(f"  objective gap   {metrics['objective_gap']:+.4f}")
    print(f"  drift err E_a   {metrics['E_a']:.4f}")
    print(f"  vol err   E_b   {metrics['E_b']:.4f}")
    print(f"  train time      {train_time:.0f}s")
    return metrics


def archive(name):
    """Tar the results folder at the repo root so one artifact comes back."""
    tar = ROOT / f"{name}.tar.gz"
    subprocess.run(["tar", "-czf", str(tar), "-C", str(ROOT), RESULTS_SUBDIR], check=True)
    print(f"\nArchived -> {tar}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--betas", type=float, nargs="+", default=[2.0, 10.0, 100.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                   help="one run per seed, each in its own results folder")
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--eps", type=float, default=1.0)
    p.add_argument("--sigma2", type=float, nargs=2, default=[10.0, 0.1])
    p.add_argument("--n-potentials", type=int, default=50)
    p.add_argument("--t-model", type=int, default=8, help="inverse-net time encoder width")
    p.add_argument("--d-model", type=int, default=32, help="inverse-net sample encoder width")
    p.add_argument("--s-init", type=float, default=0.1)
    p.add_argument("--safe-t", type=float, default=SAFE_T,
                   help="margin keeping training and evaluation away from t = 1")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-epochs", type=int, default=20000)
    p.add_argument("--min-epoch", type=int, default=20000)
    p.add_argument("--print-every", type=int, default=2000)
    p.add_argument("--n-eval", type=int, default=50000, help="pairs used for metrics A and B")
    p.add_argument("--n-times", type=int, default=21, help="quadrature nodes for C and D")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--baseline", action="store_true",
                   help="also train the LightSB-M baseline and score it against every beta")
    p.add_argument("--baseline-only", action="store_true",
                   help="train only the LightSB-M baseline, skipping LightSBB")
    p.add_argument("--archive-name", default="ground_truth")
    p.add_argument("--no-archive", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Reference J*: " + ", ".join(
        f"beta={b:g} -> {gt.solution(b, tuple(args.sigma2))['J_star']:.4f}" for b in args.betas))

    rows = []
    for seed in args.seeds:
        if not args.baseline_only:
            rows += [run_one(beta, seed, args, device) for beta in args.betas]
        if args.baseline or args.baseline_only:
            rows += run_baseline(args.betas, seed, args, device)

    # Each run already wrote its own metrics.json under its own folder, so parallel
    # runs never share an output path.
    header = (f"\n{'method':>9} {'beta':>6} {'seed':>5} {'term SW2':>9} {'plan SW2':>9} "
              f"{'cross-cov':>10} {'obj gap':>9} {'E_a':>8} {'E_b':>8}")
    print(f"\n{'=' * 84}{header}")
    for r in rows:
        print(f"{r['method']:>9} {r['beta']:>6g} {r['seed']:>5} {r['terminal_sw2']:>9.4f} "
              f"{r['plan_sw2']:>9.4f} {r['cross_cov_err']:>10.4f} {r['objective_gap']:>+9.4f} "
              f"{r['E_a']:>8.4f} {r['E_b']:>8.4f}")

    if not args.no_archive:
        archive(args.archive_name)


if __name__ == "__main__":
    main()
