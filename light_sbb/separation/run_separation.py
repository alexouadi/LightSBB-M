"""Separation tests A and B: LightSBB-M vs LightSB-M on low-entropy targets.

Sweeps the concentration delta at fixed epsilon and trains both methods at each point.
Test A collapses N(0, I2) onto N(0, diag(1, delta^2)) and is scored against the exact
Gaussian solution; Test B concentrates a 1D bimodal mixture, where no closed form
exists and the target marginal is matched empirically. Both report the distribution
metrics of the heavy-tail table alongside.

Run from inside light_sbb/:  python separation/run_separation.py --test A --device cuda:0
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ground_truth"))

import baseline_lightsb as bl  # noqa: E402
import bimodal_metrics as bm  # noqa: E402
import distribution_metrics as dm  # noqa: E402
import plan_metrics as pm  # noqa: E402
import targets as tg  # noqa: E402
from control_extraction import encode_y  # noqa: E402
from lightsbm import LightSBM, MLP_network  # noqa: E402
from train_lightsbb import training_sbb  # noqa: E402
from train_lightsbb_beta_large import training_sbb_beta_large  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS_SUBDIR = "results/separation"
LARGE_BETA = 100.0
SAFE_T = 1e-2


def run_dir(test, method, beta, delta, seed):
    """Return the folder holding one run's outputs, creating it if needed.

    Args:
        test: "A" or "B".
        method: "lightsbb" or "lightsb".
        beta: volatility penalty, absent from the baseline's folder name.
        delta: concentration parameter.
        seed: run seed.

    Returns:
        Path to results/separation/test_<t>/<family>/delta_<d>/seed_<s>/.
    """
    family = method if method == "lightsb" else f"lightsbb_beta_{beta:g}"
    path = (ROOT / RESULTS_SUBDIR / f"test_{test}" / family
            / f"delta_{delta:g}" / f"seed_{seed}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_model(dim, args, x_sampler, y_sampler, device):
    """Create a LightSBM whose potentials are initialised from both marginals."""
    model = LightSBM(dim=dim, n_potentials=args.n_potentials, epsilon=args.eps,
                     S_diagonal_init=args.s_init, is_diagonal=True).to(device)
    init_x = x_sampler.sample(args.n_potentials // 2)
    init_y = y_sampler.sample(args.n_potentials - args.n_potentials // 2)
    model.init_r_by_samples(torch.cat([init_x, init_y], dim=0))
    return model


def stages(beta, args):
    """Return the number of outer stages, which the published moderate-beta run raises.

    run_2d_benchmark.py uses K = 5 at beta = 100 but K = 15 at beta = 10, the regime
    where the inverse net has to be fitted alongside the bridge.
    """
    if args.K_moderate is not None and beta < args.large_beta:
        return args.K_moderate
    return args.K


def train_sbb(x_sampler, y_sampler, dim, beta, args, device):
    """Train LightSBB, returning (model, model_inv) with model_inv None for large beta."""
    model = build_model(dim, args, x_sampler, y_sampler, device)
    shared = dict(K=stages(beta, args), n_epochs=args.n_epochs, min_epoch=args.min_epoch,
                  batch_size=args.batch_size, lr=args.lr, eps=args.eps,
                  safe_t=args.safe_t, print_every=args.print_every, device=device)

    if beta >= args.large_beta:
        return training_sbb_beta_large(x_sampler, y_sampler, model, beta, **shared), None

    model_inv = MLP_network(input_dim=dim, t_model=args.t_model,
                            d_model=args.d_model).to(device)
    return training_sbb(x_sampler, y_sampler, model, model_inv, beta, **shared)


def train_sb(x_sampler, y_sampler, dim, args, device):
    """Train the LightSB-M baseline: the same loop at K = 1, where beta never enters."""
    model = build_model(dim, args, x_sampler, y_sampler, device)
    return training_sbb_beta_large(
        x_sampler, y_sampler, model, beta=1.0, K=1, n_epochs=args.n_epochs,
        min_epoch=args.min_epoch, batch_size=args.batch_size, lr=args.lr, eps=args.eps,
        safe_t=args.safe_t, print_every=args.print_every, device=device)


def generate_pairs(model, model_inv, x_sampler, beta, n, device, safe_t=SAFE_T):
    """Sample (X_0, X_hat_1) pairs and the Y-space endpoints the metrics need.

    Args:
        model: trained LightSBM.
        model_inv: inverse net, or None in the large-beta regime.
        x_sampler: source sampler.
        beta: volatility penalty.
        n: number of pairs.
        device: torch device.
        safe_t: margin keeping the terminal time away from the singularity at t = 1.

    Returns:
        (pairs, y_0, y_T) with pairs of shape (n, 2d).
    """
    x_0 = x_sampler.sample(n)
    y_0 = encode_y(model, model_inv, x_0, 0.0, beta, safe_t=safe_t)
    y_T = model(y_0)

    t_T = torch.full((n,), 1.0 - safe_t, device=device)
    x_1 = (y_T + model.get_drift(t_T, y_T) / beta).detach()
    return torch.cat([x_0, x_1], dim=1).cpu().numpy(), y_0, y_T


def generate_pairs_sb(model, x_sampler, n):
    """Sample (X_0, X_hat_1) for the baseline, where the Bass map is the identity."""
    y_0 = x_sampler.sample(n)
    y_T = model(y_0)
    return torch.cat([y_0, y_T], dim=1).detach().cpu().numpy(), y_0, y_T


def score(test, model, pairs, y_0, y_T, beta, delta, args, device, sb_baseline):
    """Score one trained model with the metrics its test provides.

    Test A has an exact reference, so it reuses the ground-truth metrics A-D; the SB
    baseline is scored by the module that pins its volatility to sqrt(eps) I. Test B
    has no closed form and is scored on the empirical target marginal instead. Both
    then get the per-coordinate distribution metrics of the heavy-tail table.

    Args:
        test: "A" or "B".
        model: trained LightSBM.
        pairs: (n, 2d) generated samples.
        y_0: (n, d) bridge start in Y-space.
        y_T: (n, d) bridge end in Y-space.
        beta: volatility penalty of the column being scored.
        delta: concentration parameter.
        args: parsed CLI arguments.
        device: torch device.
        sb_baseline: whether the model is the Schrodinger-bridge baseline.

    Returns:
        dict of metrics.
    """
    d = tg.dim_of(test)
    x_1 = np.asarray(pairs)[:, d:]
    # Seeded separately from training so a metrics-only rerun reproduces the sample
    # the distribution metrics are scored against.
    gen = torch.Generator(device=device).manual_seed(args.seed_for_metrics)
    target = (tg.target_sampler(test, delta, device)
              .sample(len(pairs), generator=gen).cpu().numpy())

    if test == "A":
        module = bl if sb_baseline else pm
        out = module.evaluate(model, pairs, y_0, y_T, beta, n_times=args.n_times,
                              sigma2=tuple(tg.sigma2_of(delta)), eps=args.eps,
                              seed=args.seed_for_metrics)
        # The collapse happens along the second coordinate, so name the blocks after
        # the role each axis plays rather than by index alone.
        out.update(dm.evaluate_per_coordinate(x_1, target, seed=args.seed_for_metrics,
                                              names=["free_", "collapse_"]))
        return out

    return bm.evaluate(model, pairs, y_0, y_T, beta, target, n_times=args.n_times,
                       eps=args.eps, seed=args.seed_for_metrics,
                       sb_baseline=sb_baseline, device=device)


def run_one(test, method, beta, delta, seed, args, device):
    """Train and evaluate a single (method, beta, delta, seed), returning its metric row."""
    label = "LightSB-M" if method == "lightsb" else f"LightSBB-M beta={beta:g}"
    print(f"\n{'=' * 70}\nTest {test}   {label}   delta = {delta:g}   seed = {seed}\n{'=' * 70}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    dim = tg.dim_of(test)
    x_sampler = tg.source_sampler(test, device)
    y_sampler = tg.target_sampler(test, delta, device)

    started = time.time()
    if method == "lightsb":
        model, model_inv = train_sb(x_sampler, y_sampler, dim, args, device), None
    else:
        model, model_inv = train_sbb(x_sampler, y_sampler, dim, beta, args, device)
    train_time = time.time() - started

    model.eval()
    eval_safe_t = args.eval_safe_t if args.eval_safe_t is not None else args.safe_t
    if method == "lightsb":
        pairs, y_0, y_T = generate_pairs_sb(model, x_sampler, args.n_eval)
    else:
        pairs, y_0, y_T = generate_pairs(model, model_inv, x_sampler, beta, args.n_eval,
                                         device, eval_safe_t)

    metrics = score(test, model, pairs, y_0, y_T, beta, delta, args, device,
                    sb_baseline=method == "lightsb")
    metrics.update(test=test, method=method, beta=beta, delta=delta, seed=seed,
                   K=1 if method == "lightsb" else stages(beta, args), eps=args.eps,
                   sb_lower_bound=tg.sb_lower_bound(test, delta, args.eps),
                   sbb_upper_bound=tg.sbb_upper_bound(test, delta, beta, args.eps),
                   target_entropy=tg.entropy(test, delta),
                   train_time_s=round(train_time, 1))

    out = run_dir(test, method, beta, delta, seed)
    np.save(out / "pairs.npy", pairs)
    torch.save({"model": model.state_dict(),
                "model_inv": None if model_inv is None else model_inv.state_dict(),
                "test": test, "method": method, "beta": beta, "delta": delta,
                "seed": seed, "eps": args.eps, "n_potentials": args.n_potentials},
               out / "weights.pt")
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n  " + summary_line(metrics))
    print(f"  train time      {train_time:.0f}s")
    return metrics


def summary_line(m):
    """Format the one-line per-run summary appropriate to the test."""
    if m["test"] == "A":
        return (f"plan SW2 {m['plan_sw2']:.4f}   cross-cov {m['cross_cov_err']:.4f}   "
                f"drift-only gap {m['drift_only_gap']:+.4f}   E_b {m['E_b']:.4f}   "
                f"collapse W1 {m['collapse_w1']:.4f}   KS {m['collapse_ks']:.4f}   "
                f"std {m['collapse_std_gen']:.4f} vs {m['collapse_std_true']:.4f}")
    return (f"W1 {m['w1']:.4f}   energy {m['energy']:.4f}   KS {m['ks']:.4f}   "
            f"mode err {m['mode_err']:.4f}   std {m['std_gen']:.4f} vs "
            f"{m['std_true']:.4f}   cost {m['total_cost']:.3f}   "
            f"sigma spread {m['sigma_spread']:.4f}")


def collect_rows(test):
    """Read back every metrics.json already on disk for one test.

    Args:
        test: "A" or "B".

    Returns:
        list of metric rows, ordered by (method, beta, delta, seed).
    """
    root = ROOT / RESULTS_SUBDIR / f"test_{test}"
    rows = []
    for path in sorted(root.glob("*/delta_*/seed_*/metrics.json")):
        with open(path) as f:
            rows.append(json.load(f))
    return sorted(rows, key=lambda r: (r["method"], r["beta"], r["delta"], r["seed"]))


def write_summary(test, path):
    """Write every run found on disk to one JSON file for the later figure.

    Rebuilt from the individual metrics.json files rather than from the rows of
    this invocation, so a sweep split across several devices still ends up with a
    complete summary whichever process finishes last.

    Args:
        test: "A" or "B".
        path: destination JSON file.
    """
    rows = collect_rows(test)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"test": test, "rows": rows}, f, indent=2)
    print(f"\nSummary -> {path}   ({len(rows)} runs)")


def archive(name):
    """Tar the results folder at the repo root so one artifact comes back."""
    tar = ROOT / f"{name}.tar.gz"
    subprocess.run(["tar", "-czf", str(tar), "-C", str(ROOT), RESULTS_SUBDIR], check=True)
    print(f"\nArchived -> {tar}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test", choices=["A", "B"], default="A",
                   help="A: 2D collapsing Gaussian; B: 1D bimodal mixture")
    p.add_argument("--deltas", type=float, nargs="+", default=list(tg.DELTAS))
    p.add_argument("--betas", type=float, nargs="+", default=[10.0, 100.0],
                   help="one LightSBB column per beta; >= --large-beta uses the "
                        "inverse-net-free algorithm")
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--K", type=int, default=5, help="outer bridge refinement stages")
    p.add_argument("--K-moderate", type=int, default=5,
                   help="stages used below --large-beta, where the inverse net is fitted "
                        "as well; the published 2D run uses 15 there, which triples the "
                        "training time")
    p.add_argument("--eps", type=float, default=1.0, help="diffusion scale, held fixed")
    p.add_argument("--n-potentials", type=int, default=50)
    p.add_argument("--t-model", type=int, default=8, help="inverse-net time encoder width")
    p.add_argument("--d-model", type=int, default=32, help="inverse-net sample encoder width")
    p.add_argument("--s-init", type=float, default=0.1, help="initial S diagonal")
    p.add_argument("--safe-t", type=float, default=SAFE_T,
                   help="margin keeping training away from t = 1")
    p.add_argument("--eval-safe-t", type=float, default=None,
                   help="margin used at evaluation time; defaults to --safe-t")
    p.add_argument("--large-beta", type=float, default=LARGE_BETA,
                   help="beta at or above which the inverse-net-free regime is used")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-epochs", type=int, default=20000)
    p.add_argument("--min-epoch", type=int, default=20000)
    p.add_argument("--print-every", type=int, default=2000)
    p.add_argument("--n-eval", type=int, default=50000, help="generated pairs used for metrics")
    p.add_argument("--n-times", type=int, default=21, help="quadrature nodes over [0, 1)")
    p.add_argument("--seed-for-metrics", type=int, default=0,
                   help="seed of the reference sample and the SW2 projections")
    p.add_argument("--device", default="cuda:0", help="e.g. cuda:0, cuda:1 or cpu")
    p.add_argument("--no-baseline", action="store_true", help="skip the LightSB-M column")
    p.add_argument("--baseline-only", action="store_true",
                   help="train only the LightSB-M column")
    p.add_argument("--archive-name", default=None,
                   help="archive stem; defaults to separation_test_<test>")
    p.add_argument("--no-archive", action="store_true")
    p.add_argument("--summary-only", action="store_true",
                   help="rebuild the summary from the metrics.json already on disk "
                        "and exit, training nothing")
    args = p.parse_args()

    summary_path = ROOT / RESULTS_SUBDIR / f"summary_test_{args.test}.json"
    if args.summary_only:
        write_summary(args.test, summary_path)
        return

    # An explicit --device is honoured as given; only a missing CUDA runtime
    # downgrades a cuda request, so --device cpu really means CPU.
    requested = args.device
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA unavailable, falling back to CPU (requested {requested})")
        requested = "cpu"
    device = torch.device(requested)
    print(f"Device: {device}   Test {args.test}   deltas {args.deltas}")
    print("Theory: " + ", ".join(
        f"delta={d:g} -> SB >= {tg.sb_lower_bound(args.test, d, args.eps):.2f}"
        for d in args.deltas))

    rows = []
    for seed in args.seeds:
        for delta in args.deltas:
            if not args.baseline_only:
                rows += [run_one(args.test, "lightsbb", beta, delta, seed, args, device)
                         for beta in args.betas]
            if not args.no_baseline:
                # The baseline has no beta of its own; it is trained once per
                # (delta, seed) and scored against the smallest beta of the sweep.
                rows.append(run_one(args.test, "lightsb", min(args.betas), delta, seed,
                                    args, device))

    print(f"\n{'=' * 70}")
    for r in rows:
        head = f"{r['method']:>9} beta={r['beta']:>5g} delta={r['delta']:>5g} seed={r['seed']}"
        print(f"{head}  {summary_line(r)}")

    write_summary(args.test, summary_path)
    if not args.no_archive:
        archive(args.archive_name or f"separation_test_{args.test}")


if __name__ == "__main__":
    main()
