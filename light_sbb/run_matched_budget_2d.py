"""Compare SBB with LightSB-M at equal training budget, and extract the SBB volatility.

Run from light_sbb/. Results are written after every run, finished runs are skipped, and
--tasks, --methods and --seeds let each run go to its own device:

    python run_matched_budget_2d.py --device cuda:0
    python run_matched_budget_2d.py --tasks M8G --methods sbb --seeds 0 --device cuda:1
"""

import argparse
import csv
import json
import os
import sys
import tarfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

LIGHT_SBB = Path(__file__).resolve().parent
sys.path.insert(0, str(LIGHT_SBB))
sys.path.insert(0, str(LIGHT_SBB / "ground_truth"))

from control_extraction import bridge_y, map_x, sigma_x  # noqa: E402
from lightsbm import LightSBM  # noqa: E402
from train_lightsbb_beta_large import training_sbb_beta_large  # noqa: E402
from utils import GeneratorTwoD, wasserstein  # noqa: E402

ROOT = LIGHT_SBB.parent
RESULTS_DIR = ROOT / "results" / "matched_budget_2d"
ARCHIVE = ROOT / "matched_budget_2d.tar.gz"

# Settings of run_2d_benchmark.py: name -> (source, target, eps, beta, K).
TASKS = {
    "M8G": ("moons", "8gaussians", 5.0, 100.0, 5),
    "NM": ("normal", "moons", 1.0, 100.0, 5),
}
METHODS = ("sbb", "sb", "sb_matched", "sb_beta_inf")
# The Table 2 recipe for LightSB-M: one stage at a huge beta, no Bass map at inference.
BETA_INF = 1e5
DIM = 2
N_POTENTIALS = 50
S_INIT = 0.1
BATCH_SIZE = 512
LR = 1e-3
SAFE_T = 1e-2
STAGE_EPOCHS = 20000
SAMPLING_BATCH = 4096
N_EVAL = 10000
EVAL_SEED_OFFSET = 10000
PROFILE_TIMES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.99)
N_PATH = 5000
SUMMARY_FIELDS = ["task", "method", "seed", "w2", "n_gradient_steps", "train_seconds",
                  "eig_min_on_path", "eig_max_on_path", "frac_eig_negative_on_path",
                  "state_std_t99", "s_max_min_coordinate", "s_threshold"]


def parse_args():
    """Parse the runs to launch and the device."""
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=sorted(TASKS))
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--print-every", type=int, default=5000)
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


def train(task, method, seed, source, target, device, print_every):
    """Train one method with the settings of run_2d_benchmark.py.

    SBB runs K stages. The Schrodinger bridge is the K = 1 stage of the same loop, where
    beta is never read, run for one stage or for as many steps as all the SBB stages.

    Args:
        task: Key of ``TASKS``.
        method: One of ``METHODS``.
        seed: Random seed.
        source: Source sampler.
        target: Target sampler.
        device: Training device.
        print_every: Epochs between progress prints.

    Returns:
        Tuple ``(model, n_gradient_steps, train_seconds)``.
    """
    _, _, eps, beta, K = TASKS[task]
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = LightSBM(dim=DIM, n_potentials=N_POTENTIALS, epsilon=eps, S_diagonal_init=S_INIT,
                     is_diagonal=True, sampling_batch_size=SAMPLING_BATCH).to(device)
    # The moons generator samples on CPU whatever the device it was given.
    model.init_r_by_samples(torch.cat([source.sample(N_POTENTIALS // 2).to(device),
                                       target.sample(N_POTENTIALS - N_POTENTIALS // 2).to(device)]))

    stages = K if method == "sbb" else 1
    epochs = K * STAGE_EPOCHS if method == "sb_matched" else STAGE_EPOCHS
    train_beta = {"sbb": beta, "sb_beta_inf": BETA_INF}.get(method, 1.0)
    start = time.perf_counter()
    model = training_sbb_beta_large(
        source, target, model, train_beta, K=stages, n_epochs=epochs,
        min_epoch=epochs, batch_size=BATCH_SIZE, lr=LR, eps=eps, safe_t=SAFE_T,
        print_every=print_every, device=device,
    )
    return model, stages * epochs, time.perf_counter() - start


def terminal_samples(model, method, beta, x_0):
    """Transport source samples, as in run_2d_benchmark.py.

    Args:
        model: Trained ``LightSBM``.
        method: One of ``METHODS``.
        beta: Volatility penalty.
        x_0: (n, DIM) source samples.

    Returns:
        Tuple of the (n, DIM) tensors ``(y_0, y_T, x_T)``.
    """
    if method != "sbb":
        y_T = model(x_0)
        return x_0, y_T, y_T

    t_zeros = torch.zeros(len(x_0), device=x_0.device)
    y_0 = (x_0 - model.get_drift(t_zeros, x_0) / beta).detach()
    y_T = model(y_0)
    t_end = torch.full((len(y_T),), 1.0 - SAFE_T, device=y_T.device)
    return y_0, y_T, (y_T + model.get_drift(t_end, y_T) / beta).detach()


def volatility_profile(model, beta, eps, y_0, y_T):
    """Evaluate the volatility along the learned bridge, normalized by sqrt(eps).

    Args:
        model: Trained ``LightSBM``.
        beta: Volatility penalty.
        eps: Diffusion scale.
        y_0: (n, DIM) bridge starts.
        y_T: (n, DIM) bridge ends.

    Returns:
        Tuple of the arrays to save and the scalar diagnostics. A value of 1 everywhere is
        the constant volatility of the Schrodinger bridge, and a negative eigenvalue means
        the map from Y to X folds.
    """
    eps_t = torch.as_tensor(eps, dtype=y_0.dtype, device=y_0.device)
    positions, eigenvalues = [], []
    for t in PROFILE_TIMES:
        y_t = bridge_y(y_0[:N_PATH], y_T[:N_PATH], t, eps_t)
        t_vec = torch.full((len(y_t),), t, dtype=y_t.dtype, device=y_t.device)
        positions.append(map_x(model, t_vec, y_t, beta).detach().cpu().numpy())
        sigma = sigma_x(model, t_vec, y_t, beta).detach() / np.sqrt(eps)
        eigenvalues.append(torch.linalg.eigvalsh(sigma).cpu().numpy())

    eigenvalues = np.stack(eigenvalues)
    diagnostics = {
        "eig_min_on_path": float(eigenvalues.min()),
        "eig_max_on_path": float(eigenvalues.max()),
        "frac_eig_negative_on_path": float((eigenvalues < 0).mean()),
        "frac_eig_negative_by_time": (eigenvalues < 0).mean(axis=(1, 2)).tolist(),
        "state_std_by_time": eigenvalues.std(axis=1).mean(axis=1).tolist(),
        "state_std_t99": float(eigenvalues[-1].std(axis=0).mean()),
    }
    arrays = {"times": np.array(PROFILE_TIMES), "x_t": np.stack(positions),
              "eigenvalues": eigenvalues}
    return arrays, diagnostics


def run_one(task, method, seed, device, print_every):
    """Train and score one run, then refresh the summary and the archive.

    A failed run is recorded and the others go on.

    Args:
        task: Key of ``TASKS``.
        method: One of ``METHODS``.
        seed: Random seed.
        device: Device of this run.
        print_every: Epochs between progress prints.
    """
    run_dir = RESULTS_DIR / task / method / f"seed_{seed}"
    if (run_dir / "metrics.json").exists():
        print(f"{task} {method} seed={seed}: already complete, skipping", flush=True)
        return
    print(f"{task} {method} seed={seed}: starting on {device}", flush=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    source_name, target_name, eps, beta, K = TASKS[task]
    source = GeneratorTwoD(source_name, DIM, device)
    target = GeneratorTwoD(target_name, DIM, device)

    try:
        model, n_steps, train_seconds = train(task, method, seed, source, target, device, print_every)
        write_atomic(run_dir / "weights.pt", lambda tmp: torch.save(model.state_dict(), tmp))

        # Same evaluation samples for every method of a seed, so the W2 gap is not sampling noise.
        torch.manual_seed(EVAL_SEED_OFFSET + seed)
        np.random.seed(EVAL_SEED_OFFSET + seed)
        x_0 = source.sample(N_EVAL).to(device)
        x_target = target.sample(N_EVAL).to(device)
        y_0, y_T, x_T = terminal_samples(model, method, beta, x_0)
        if not torch.isfinite(x_T).all():
            raise ValueError("non-finite terminal samples")

        s_max = model.S_log_diagonal_matrix.exp().max(dim=0).values
        metrics = {"task": task, "method": method, "seed": seed, "eps": eps,
                   "beta": BETA_INF if method == "sb_beta_inf" else beta, "K": K,
                   "w2": float(wasserstein(x_T, x_target)), "n_gradient_steps": n_steps,
                   "train_seconds": train_seconds,
                   "s_max_min_coordinate": float(s_max.min())}
        np.save(run_dir / "x_T.npy", x_T.detach().cpu().numpy())

        if method == "sbb":
            # T_beta^+[phi] is -infinity below this value, at T = 1 (Proposition A.1).
            metrics["s_threshold"] = 1.0 / (beta + 1.0)
            arrays, diagnostics = volatility_profile(model, beta, eps, y_0, y_T)
            metrics.update(diagnostics)
            np.savez(run_dir / "profile.npz", **arrays)

        write_atomic(run_dir / "metrics.json",
                     lambda tmp: Path(tmp).write_text(json.dumps(metrics, indent=2)))
        (run_dir / "failure.json").unlink(missing_ok=True)
    except (RuntimeError, ValueError) as error:
        print(f"{task} {method} seed={seed}: failed with {error}", flush=True)
        (run_dir / "failure.json").write_text(json.dumps(
            {"task": task, "method": method, "seed": seed, "error": str(error),
             "traceback": traceback.format_exc()}, indent=2))

    write_summary()
    write_archive()


def write_summary():
    """Rebuild the summary table from every finished run."""
    rows = [json.loads(path.read_text())
            for path in sorted(RESULTS_DIR.glob("*/*/seed_*/metrics.json"))]

    def write(tmp):
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, SUMMARY_FIELDS, extrasaction="ignore")
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
    """Run every requested (task, method, seed) in turn on the chosen device."""
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    for task in args.tasks:
        for method in args.methods:
            for seed in args.seeds:
                run_one(task, method, seed, device, args.print_every)
    print(f"archive: {ARCHIVE}", flush=True)


if __name__ == "__main__":
    main()
