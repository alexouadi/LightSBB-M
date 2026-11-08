"""Train, diagnose and score the admissible LightSBB variant on the ALAE latents.

Run from light_sbb/, one process per device. Each finished run refreshes the archive:

    python run_alae_admissible.py --seeds 0 --device cuda:0
"""

import argparse
import json
import os
import pickle
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

LIGHT_SBB = Path(__file__).resolve().parent
sys.path.insert(0, str(LIGHT_SBB))
sys.path.insert(0, str(LIGHT_SBB / "alae"))

from baselines.alae_data import DIM, MODEL_DIR, ROOT, load_alae_splits  # noqa: E402
from baselines.diagnose_alae_invertibility import (curvature_along_path,  # noqa: E402
                                                   summarize, weight_diagnostics)
from lightsbm import LightSBM  # noqa: E402
from run_alae_j_sweep import prepare_shared_images, write_atomic  # noqa: E402
from train_lightsbb_admissible import (bass_map, invert_bass_map,  # noqa: E402
                                       training_sbb_admissible)

RESULTS = ROOT / "results" / "alae_admissible"
CHECKPOINTS = MODEL_DIR / "admissible"
ARCHIVE = ROOT / "alae_admissible.tar.gz"


def parse_args():
    """Parse the training and evaluation settings."""
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--J", type=int, default=10)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--n-epochs", type=int, default=15000)
    p.add_argument("--min-epoch", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--damping", type=float, default=0.2)
    p.add_argument("--margin", type=float, default=0.05)
    p.add_argument("--s-init", type=float, default=1.0)
    p.add_argument("--safe-t", type=float, default=1e-2)
    p.add_argument("--times", type=float, nargs="+", default=[0.0, 0.5, 0.9, 0.99])
    p.add_argument("--n-points", type=int, default=256)
    p.add_argument("--n-inverse", type=int, default=2000)
    p.add_argument("--jacobian-batch", type=int, default=64)
    p.add_argument("--print-every", type=int, default=4000)
    p.add_argument("--n-images", type=int, default=1000)
    p.add_argument("--n-reference", type=int, default=1000)
    p.add_argument("--decode-batch", type=int, default=32)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--no-fid", action="store_true")
    return p.parse_args()


def run_name(args, seed):
    """Return the folder name of a run."""
    return f"b{args.beta:g}_K{args.k}_w{args.damping:g}_m{args.margin:g}_s{seed}"


def transport(model, x_0, args):
    """Map source latents to the target with the exact inverse and the Bass map.

    Args:
        model: Trained ``LightSBM``.
        x_0: (n, DIM) source latents on the model's device.
        args: Parsed arguments.

    Returns:
        (n, DIM) transported latents on the model's device.
    """
    y_0, _ = invert_bass_map(model, x_0, 0.0, args.beta)
    with torch.no_grad():
        y_T = model(y_0)
    return bass_map(model.double(), y_T.double(), 1.0 - args.safe_t, args.beta).float(), model.float()


def diagnostics(model, splits, args, device):
    """Measure the curvature and the solver residuals on held-out latents.

    Args:
        model: Trained ``LightSBM``.
        splits: Output of ``load_alae_splits``.
        args: Parsed arguments.
        device: Device of the model.

    Returns:
        JSON-ready dict.
    """
    adults = torch.as_tensor(splits["test_latents"][splits["x_inds_test"]][:args.n_inverse], device=device)
    children = torch.as_tensor(splits["test_latents"][splits["y_inds_test"]][:args.n_inverse], device=device)

    y_0, residual_0 = invert_bass_map(model, adults, 0.0, args.beta)
    _, residual_T = invert_bass_map(model, children, 1.0 - args.safe_t, args.beta)
    with torch.no_grad():
        y_T = model(y_0)

    precise = model.double()
    results = {
        "epsilon": float(precise.epsilon),
        "weights": weight_diagnostics(precise, y_0.double(), args.beta),
        "inverse_residual_t0": summarize((residual_0 / adults.norm(dim=1)).cpu().numpy()),
        "inverse_residual_terminal": summarize((residual_T / children.norm(dim=1)).cpu().numpy()),
        "curvature": curvature_along_path(precise, y_0[:args.n_points].double(),
                                          y_T[:args.n_points].double(), args),
    }
    model.float()
    return results


def write_archive():
    """Bundle every finished run into one archive at the repo root."""

    def write(tmp):
        with tarfile.open(tmp, "w:gz") as tar:
            for folder in (RESULTS, CHECKPOINTS):
                if folder.is_dir():
                    tar.add(folder, arcname=folder.relative_to(ROOT).as_posix())

    write_atomic(ARCHIVE, write)


def run_one(seed, args, device):
    """Train, diagnose and score one seed.

    Args:
        seed: Random seed.
        args: Parsed arguments.
        device: Training device.
    """
    from baselines.eval_alae_baseline import load_source_latents, score_transport

    torch.manual_seed(seed)
    np.random.seed(seed)
    splits = load_alae_splits(args.data_dir)
    x_pool = torch.as_tensor(splits["x_train"], device=device)
    y_pool = torch.as_tensor(splits["y_train"], device=device)

    model = LightSBM(dim=DIM, n_potentials=args.J, epsilon=args.eps, S_diagonal_init=args.s_init,
                     is_diagonal=True, sampling_batch_size=256).to(device)
    means = torch.cat([x_pool[torch.randperm(len(x_pool))[:args.J // 2]],
                       y_pool[torch.randperm(len(y_pool))[:args.J - args.J // 2]]])
    model.init_r_by_samples(means)

    start = time.perf_counter()
    model, residuals = training_sbb_admissible(
        x_pool, y_pool, model, args.beta, K=args.k, lr=args.lr, n_epochs=args.n_epochs,
        min_epoch=args.min_epoch, batch_size=args.batch_size, eps=args.eps, safe_t=args.safe_t,
        damping=args.damping, margin=args.margin, print_every=args.print_every, device=device)
    train_seconds = time.perf_counter() - start

    name = run_name(args, seed)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    with open(CHECKPOINTS / f"{name}.pkl", "wb") as f:
        pickle.dump({"state_dict": state, **{k: v for k, v in vars(args).items() if k != "seeds"}}, f)

    results = {"train_seconds": train_seconds, "solver_residual_per_stage": residuals,
               "diagnostics": diagnostics(model, splits, args, device)}

    run_args = argparse.Namespace(**vars(args), model="lightsbb", seed=seed, n_steps=None)
    torch.manual_seed(seed)
    x_0 = load_source_latents(run_args)
    x_1, model = transport(model, x_0.to(device), args)
    if not torch.isfinite(x_1).all():
        raise ValueError("non-finite transported latents")
    prepare_shared_images(run_args, x_0, device)
    metrics_path = Path(score_transport(run_args, f"admissible/{name}", x_0, x_1.cpu(),
                                        {"beta": args.beta, "J": args.J}, device))
    results["metrics"] = json.loads(metrics_path.read_text())

    out_dir = RESULTS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"finished {name}: {results['metrics']}", flush=True)


def main():
    """Run every requested seed and refresh the archive after each one."""
    args = parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    for seed in args.seeds:
        run_one(seed, args, device)
        write_archive()


if __name__ == "__main__":
    main()
