"""Rerun the 2D comparison with the budgets and training loops of the March 2026 benchmark.

Run from light_sbb/. Means are always initialized from samples. Evaluation is the one of
run_matched_budget_2d.py, with the same samples per seed, so the numbers compare directly:

    python run_2d_legacy.py --seeds 0 1 --device cuda:0
"""

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

LIGHT_SBB = Path(__file__).resolve().parent
sys.path.insert(0, str(LIGHT_SBB))
sys.path.insert(0, str(LIGHT_SBB / "ground_truth"))

from lightsbm import LightSBM  # noqa: E402
from run_matched_budget_2d import (BATCH_SIZE, DIM, EVAL_SEED_OFFSET, LR, N_EVAL, N_POTENTIALS,  # noqa: E402
                                   S_INIT, SAFE_T, SAMPLING_BATCH, TASKS, terminal_samples, write_atomic)
from train_lightsbb_beta_large import training_sbb_beta_large  # noqa: E402
from utils import GeneratorTwoD, wasserstein  # noqa: E402

ROOT = LIGHT_SBB.parent
RESULTS_DIR = ROOT / "results" / "legacy_2d"
ARCHIVE = ROOT / "legacy_2d.tar.gz"

# Last commit before gradient clipping and the stop-gradient on the map were added.
OLD_LOOP_REV = "a0c1808"
LEGACY_K = {"M8G": 5, "NM": 25}
BASE_EPOCHS = 10000
MIN_EPOCHS = 5000
BETA_INF = 1e5
# name -> (loop, method). "matched" is one stage at the step count of all the SBB stages.
CONFIGS = {
    "current_sb": ("current", "sb"),
    "current_sb_matched": ("current", "matched"),
    "current_sbb": ("current", "sbb"),
    "old_sb": ("old", "sb"),
    "old_sb_matched": ("old", "matched"),
    "old_sbb": ("old", "sbb"),
}
SUMMARY_FIELDS = ["task", "config", "seed", "w2", "n_gradient_steps", "train_seconds"]


def parse_args():
    """Parse the runs to launch and the device."""
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", choices=sorted(LEGACY_K), default=sorted(LEGACY_K))
    p.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def stage_epochs(k):
    """Return the epochs of outer stage ``k`` under the March 2026 schedule."""
    return max(MIN_EPOCHS, int(BASE_EPOCHS * np.exp(-0.2 * k)))


def load_old_loop():
    """Import the training loop as it stood at ``OLD_LOOP_REV``, from git history.

    Returns:
        The ``training_sbb_beta_large`` function of that commit.
    """
    source = subprocess.run(["git", "show", f"{OLD_LOOP_REV}:light_sbb/train_lightsbb_beta_large.py"],
                            cwd=ROOT, check=True, capture_output=True, text=True).stdout
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "old_loop.py"
        path.write_text(source, encoding="utf8")
        spec = importlib.util.spec_from_file_location("old_loop", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.training_sbb_beta_large


def train(task, config, seed, source, target, device):
    """Train one configuration from means initialized on samples.

    Args:
        task: Key of ``LEGACY_K``.
        config: Key of ``CONFIGS``.
        seed: Random seed.
        source: Source sampler.
        target: Target sampler.
        device: Training device.

    Returns:
        Tuple ``(model, n_gradient_steps, train_seconds)``.
    """
    loop, method = CONFIGS[config]
    _, _, eps, beta, _ = TASKS[task]
    sbb_steps = sum(stage_epochs(k) for k in range(LEGACY_K[task]))
    stages = LEGACY_K[task] if method == "sbb" else 1
    epochs = sbb_steps if method == "matched" else BASE_EPOCHS
    steps = sbb_steps if method != "sb" else BASE_EPOCHS

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = LightSBM(dim=DIM, n_potentials=N_POTENTIALS, epsilon=eps, S_diagonal_init=S_INIT,
                     is_diagonal=True, sampling_batch_size=SAMPLING_BATCH).to(device)
    model.init_r_by_samples(torch.cat([source.sample(N_POTENTIALS // 2).to(device),
                                       target.sample(N_POTENTIALS - N_POTENTIALS // 2).to(device)]))

    train_beta = beta if method == "sbb" else BETA_INF
    settings = dict(K=stages, n_epochs=epochs, batch_size=BATCH_SIZE, lr=LR, eps=eps, safe_t=SAFE_T, device=device)
    start = time.perf_counter()
    if loop == "old":
        model = load_old_loop()(source, target, model, train_beta, **settings)
    else:
        model = training_sbb_beta_large(source, target, model, train_beta, min_epoch=MIN_EPOCHS, **settings)
    return model, steps, time.perf_counter() - start


def run_one(task, config, seed, device):
    """Train and score one run, then refresh the summary and the archive.

    Args:
        task: Key of ``LEGACY_K``.
        config: Key of ``CONFIGS``.
        seed: Random seed.
        device: Device of this run.
    """
    run_dir = RESULTS_DIR / task / config / f"seed_{seed}"
    if (run_dir / "metrics.json").exists():
        print(f"{task} {config} seed={seed}: already complete, skipping", flush=True)
        return
    print(f"{task} {config} seed={seed}: starting on {device}", flush=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    source_name, target_name, eps, beta, _ = TASKS[task]
    source = GeneratorTwoD(source_name, DIM, device)
    target = GeneratorTwoD(target_name, DIM, device)
    method = CONFIGS[config][1]

    try:
        model, n_steps, train_seconds = train(task, config, seed, source, target, device)
        write_atomic(run_dir / "weights.pt", lambda tmp: torch.save(model.state_dict(), tmp))

        torch.manual_seed(EVAL_SEED_OFFSET + seed)
        np.random.seed(EVAL_SEED_OFFSET + seed)
        x_0 = source.sample(N_EVAL).to(device)
        x_target = target.sample(N_EVAL).to(device)
        _, _, x_T = terminal_samples(model, "sbb" if method == "sbb" else "sb", beta, x_0)
        x_T = x_T.detach()
        if not torch.isfinite(x_T).all():
            raise ValueError("non-finite terminal samples")

        metrics = {"task": task, "config": config, "seed": seed, "eps": eps,
                   "beta": beta if method == "sbb" else BETA_INF,
                   "K": LEGACY_K[task] if method == "sbb" else 1, "w2": float(wasserstein(x_T, x_target)),
                   "n_gradient_steps": n_steps, "train_seconds": train_seconds}
        np.save(run_dir / "x_T.npy", x_T.cpu().numpy())
        write_atomic(run_dir / "metrics.json", lambda tmp: Path(tmp).write_text(json.dumps(metrics, indent=2)))
        (run_dir / "failure.json").unlink(missing_ok=True)
    except (RuntimeError, ValueError) as error:
        print(f"{task} {config} seed={seed}: failed with {error}", flush=True)
        (run_dir / "failure.json").write_text(json.dumps(
            {"task": task, "config": config, "seed": seed, "error": str(error),
             "traceback": traceback.format_exc()}, indent=2))

    write_summary()
    write_archive()


def write_summary():
    """Rebuild the summary table from every finished run."""
    rows = [json.loads(path.read_text()) for path in sorted(RESULTS_DIR.glob("*/*/seed_*/metrics.json"))]

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
    """Run every requested (task, config, seed) in turn on the chosen device."""
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    for task in args.tasks:
        for config in args.configs:
            for seed in args.seeds:
                run_one(task, config, seed, device)
    print(f"archive: {ARCHIVE}", flush=True)


if __name__ == "__main__":
    main()
