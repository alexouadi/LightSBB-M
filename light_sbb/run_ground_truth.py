"""Ground-truth validation of the SBB plan on the analytic 2D Gaussian benchmark.

Trains LightSBB on N(0, I2) -> N(0, diag(10, 0.1)) for each beta, then reports metrics
A-D against the exact solution: plan SW2, cross-covariance error, objective gap, and
drift/volatility recovery. Answers the meta-review objection that only the terminal
marginal was ever measured.

Run from inside light_sbb/:  python run_ground_truth.py
Results, weights and (X_0, X_1) pairs are archived to a single .tar.gz at the repo root.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

import gaussian_ground_truth as gt
import plan_metrics as pm
from control_extraction import encode_y
from lightsbm import LightSBM, MLP_network
from train_lightsbb import training_sbb
from train_lightsbb_beta_large import training_sbb_beta_large

ROOT = Path(__file__).resolve().parents[1]
RESULTS_SUBDIR = "results/ground_truth"
LARGE_BETA = 100.0
SAFE_T = 1e-2


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
                  safe_t=SAFE_T, print_every=args.print_every, device=device)

    if beta >= LARGE_BETA:
        return training_sbb_beta_large(x_sampler, y_sampler, model, beta, **shared), None

    model_inv = MLP_network(input_dim=2, t_model=8, d_model=32).to(device)
    return training_sbb(x_sampler, y_sampler, model, model_inv, beta, **shared)


def generate_pairs(model, model_inv, x_sampler, beta, n, device):
    """Sample (X_0, X_hat_1) pairs and the Y-space endpoints the metrics need.

    Args:
        model: trained LightSBM.
        model_inv: inverse net, or None in the large-beta regime.
        x_sampler: source sampler.
        beta: SBB volatility penalty.
        n: number of pairs.
        device: torch device.

    Returns:
        (pairs, y_0, y_T) with pairs of shape (n, 4).
    """
    x_0 = x_sampler.sample(n)
    y_0 = encode_y(model, model_inv, x_0, 0.0, beta, safe_t=SAFE_T)
    y_T = model(y_0)

    t_T = torch.full((n,), 1.0 - SAFE_T, device=device)
    x_1 = (y_T + model.get_drift(t_T, y_T) / beta).detach()
    return torch.cat([x_0, x_1], dim=1).cpu().numpy(), y_0, y_T


def run_one(beta, args, device):
    """Train and evaluate a single beta, returning its metric row."""
    print(f"\n{'=' * 70}\nbeta = {beta:g}   seed = {args.seed}   K = {args.K}\n{'=' * 70}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x_sampler = GaussianSampler([1.0, 1.0], device)
    y_sampler = GaussianSampler(args.sigma2, device)

    started = time.time()
    model, model_inv = train(x_sampler, y_sampler, beta, args, device)
    train_time = time.time() - started

    model.eval()
    pairs, y_0, y_T = generate_pairs(model, model_inv, x_sampler, beta, args.n_eval, device)

    metrics = pm.evaluate(model, pairs, y_0, y_T, beta, n_times=args.n_times,
                          sigma2=tuple(args.sigma2), eps=args.eps, seed=args.seed)
    metrics.update(beta=beta, seed=args.seed, K=args.K, train_time_s=round(train_time, 1))

    out = ROOT / RESULTS_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    stem = f"b{beta:g}_K{args.K}_seed{args.seed}"
    np.save(out / f"pairs_{stem}.npy", pairs)
    torch.save({"model": model.state_dict(),
                "model_inv": None if model_inv is None else model_inv.state_dict(),
                "beta": beta, "K": args.K, "seed": args.seed, "eps": args.eps},
               out / f"weights_{stem}.pt")

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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--eps", type=float, default=1.0)
    p.add_argument("--sigma2", type=float, nargs=2, default=[10.0, 0.1])
    p.add_argument("--n-potentials", type=int, default=50)
    p.add_argument("--s-init", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-epochs", type=int, default=20000)
    p.add_argument("--min-epoch", type=int, default=20000)
    p.add_argument("--print-every", type=int, default=2000)
    p.add_argument("--n-eval", type=int, default=50000, help="pairs used for metrics A and B")
    p.add_argument("--n-times", type=int, default=21, help="quadrature nodes for C and D")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--archive-name", default="ground_truth")
    p.add_argument("--no-archive", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Reference J*: " + ", ".join(
        f"beta={b:g} -> {gt.solution(b, tuple(args.sigma2))['J_star']:.4f}" for b in args.betas))

    rows = [run_one(beta, args, device) for beta in args.betas]

    out = ROOT / RESULTS_SUBDIR
    with open(out / f"metrics_K{args.K}_seed{args.seed}.json", "w") as f:
        json.dump(rows, f, indent=2)

    header = f"\n{'beta':>6} {'term SW2':>9} {'plan SW2':>9} {'cross-cov':>10} {'obj gap':>9} {'E_a':>8} {'E_b':>8}"
    print(f"\n{'=' * 70}{header}")
    for r in rows:
        print(f"{r['beta']:>6g} {r['terminal_sw2']:>9.4f} {r['plan_sw2']:>9.4f} "
              f"{r['cross_cov_err']:>10.4f} {r['objective_gap']:>+9.4f} "
              f"{r['E_a']:>8.4f} {r['E_b']:>8.4f}")

    if not args.no_archive:
        archive(args.archive_name)


if __name__ == "__main__":
    main()
