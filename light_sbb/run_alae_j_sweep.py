"""Sweep the number of Gaussian-mixture components J on the ALAE latent space.

Run from light_sbb/. Every run writes its results as soon as it finishes, and a run that
is already complete is skipped, so a sweep can be split across devices or resumed:

    python run_alae_j_sweep.py --js 1 5 --seeds 0 1 2 --device cuda:0
    python run_alae_j_sweep.py --js 10 --seeds 0 --device cuda:1
"""

import argparse
import csv
import json
import os
import pickle
import sys
import tarfile
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

LIGHT_SBB = Path(__file__).resolve().parent
sys.path.insert(0, str(LIGHT_SBB))
# ALAE's own modules import each other flatly, so its package directory has to be on the path.
sys.path.insert(0, str(LIGHT_SBB / "alae"))

from baselines.alae_data import DIM, MODEL_DIR, ROOT, load_alae_splits  # noqa: E402
from lightsbm import LightSBM, MLP_network  # noqa: E402
from train_lightsbb import training_sbb  # noqa: E402
from utils import TensorSampler  # noqa: E402

SWEEPS = {"random": "j_sweep", "samples": "j_sweep_r_samples"}
DEFAULT_EPS = 0.1
T_MODEL = 32
D_MODEL = 128
SUMMARY_FIELDS = ["J", "seed", "pct_age_leq_18", "avg_age", "std_age", "identity_sim_mean",
                  "identity_sim_std", "fid", "n_faces_detected", "n_parameters", "train_seconds",
                  "transport_seconds", "peak_train_memory_gb"]


def parse_args():
    """Parse the sweep grid, the device and the training settings."""
    p = argparse.ArgumentParser()
    p.add_argument("--js", type=int, nargs="+", default=[1, 5, 10, 50, 100])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=DEFAULT_EPS)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n-epochs", type=int, default=15000)
    p.add_argument("--min-epoch", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--s-init", type=float, default=0.1)
    p.add_argument("--r-init", choices=sorted(SWEEPS), default="random",
                   help="initial mixture means; each choice has its own results folder and archive")
    p.add_argument("--no-grad-clip", action="store_true",
                   help="train without gradient clipping, as the original ALAE checkpoints were")
    p.add_argument("--safe-t", type=float, default=1e-2)
    p.add_argument("--print-every", type=int, default=4000)
    p.add_argument("--n-images", type=int, default=1000)
    p.add_argument("--n-reference", type=int, default=1000)
    p.add_argument("--decode-batch", type=int, default=32)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--no-fid", action="store_true")
    p.add_argument("--keep-images", action="store_true",
                   help="keep the decoded outputs, about 1.5 GB per run")
    return p.parse_args()


def sweep_name(args):
    """Return the folder name of this sweep, so settings never share or skip each other's runs."""
    name = SWEEPS[args.r_init]
    if args.eps != DEFAULT_EPS:
        name += f"_e{args.eps:g}"
    if args.no_grad_clip:
        name += "_noclip"
    return name


def results_dir(args):
    """Return the results folder of this sweep."""
    return ROOT / "results" / "alae_baselines" / sweep_name(args)


def checkpoint_dir(args):
    """Return the checkpoint folder of this sweep."""
    return MODEL_DIR / sweep_name(args)


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


@contextmanager
def shared_lock(path):
    """Hold an exclusive lock, which the system releases if the process dies.

    Args:
        path: Lock file, created if missing.
    """
    import fcntl  # Linux only, and only reached on the machine running the evaluation.

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def peak_memory_gb(device):
    """Return the peak GPU memory of this process in GB, or None on CPU."""
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 1024 ** 3


def train_run(J, seed, args, device, checkpoint_path):
    """Train one (J, seed) run and save its checkpoint.

    Args:
        J: Number of mixture components.
        seed: Random seed.
        args: Parsed CLI arguments.
        device: Training device.
        checkpoint_path: Destination of the pickled checkpoint.

    Returns:
        The checkpoint dict.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    splits = load_alae_splits(args.data_dir)
    x_sampler = TensorSampler(torch.tensor(splits["x_train"]), device=device)
    y_sampler = TensorSampler(torch.tensor(splits["y_train"]), device=device)

    model = LightSBM(dim=DIM, n_potentials=J, epsilon=args.eps, S_diagonal_init=args.s_init,
                     is_diagonal=True).to(device)
    if args.r_init == "samples":
        model.init_r_by_samples(torch.cat([x_sampler.sample(J // 2), y_sampler.sample(J - J // 2)], dim=0))
    model_inv = MLP_network(input_dim=DIM, t_model=T_MODEL, d_model=D_MODEL).to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    model, model_inv = training_sbb(
        x_sampler, y_sampler, model, model_inv, args.beta, K=args.k, lr=args.lr,
        n_epochs=args.n_epochs, min_epoch=args.min_epoch, batch_size=args.batch_size,
        eps=args.eps, safe_t=args.safe_t, print_every=args.print_every, device=device,
    )

    checkpoint = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "state_dict_inv": {k: v.cpu() for k, v in model_inv.state_dict().items()},
        "J": J,
        "seed": seed,
        "beta": args.beta,
        "eps": args.eps,
        "k": args.k,
        "n_epochs": args.n_epochs,
        "min_epoch": args.min_epoch,
        "s_init": args.s_init,
        "r_init": args.r_init,
        "grad_clip": not args.no_grad_clip,
        "safe_t": args.safe_t,
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "train_seconds": time.perf_counter() - start,
        "peak_train_memory_gb": peak_memory_gb(device),
    }

    def dump(tmp):
        with open(tmp, "wb") as f:
            pickle.dump(checkpoint, f)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(checkpoint_path, dump)
    return checkpoint


def load_models(checkpoint, device):
    """Rebuild the bridge and its inverse network from a checkpoint.

    Args:
        checkpoint: Checkpoint dict written by ``train_run``.
        device: Device the models are moved to.

    Returns:
        Tuple ``(model, model_inv)`` in evaluation mode.
    """
    model = LightSBM(dim=DIM, n_potentials=checkpoint["J"], epsilon=checkpoint["eps"],
                     S_diagonal_init=checkpoint["s_init"], is_diagonal=True)
    model.load_state_dict(checkpoint["state_dict"])
    model_inv = MLP_network(input_dim=DIM, t_model=T_MODEL, d_model=D_MODEL)
    model_inv.load_state_dict(checkpoint["state_dict_inv"])
    return model.to(device).eval(), model_inv.to(device).eval()


def transport(model, model_inv, x_0, beta, safe_t):
    """Map source latents to the target through Y-space and the Bass map.

    Args:
        model: Trained ``LightSBM``.
        model_inv: Trained inverse network.
        x_0: (n, DIM) source latents on the models' device.
        beta: Volatility weight of the run.
        safe_t: Margin at which the Bass map is evaluated.

    Returns:
        (n, DIM) transported latents on the models' device.
    """
    device = x_0.device
    with torch.no_grad():
        y_0 = model_inv(torch.zeros((len(x_0), 1), device=device), x_0)
        y_mapped = model(y_0)
    # get_drift differentiates through its own input, so it runs with grad enabled.
    t_end = torch.full((len(y_mapped),), 1.0 - safe_t, device=device)
    return (y_mapped + model.get_drift(t_end, y_mapped) / beta).detach()


def prepare_shared_images(args, x_0, device):
    """Decode the input and reference images once, even with runs in parallel.

    Args:
        args: Parsed CLI arguments.
        x_0: (n, DIM) source latents on CPU.
        device: Device the ALAE decoder runs on.
    """
    from alae.alae_ffhq_inference import load_model
    from baselines.eval_alae_baseline import (ALAE_ARTIFACTS, ALAE_CONFIG, IMAGE_DIR,
                                              decode_to_folder)
    from baselines.fid_alae import reference_folder

    input_folder = IMAGE_DIR / f"input_n{args.n_images}"
    with shared_lock(IMAGE_DIR / ".j_sweep.lock"):
        alae_model = load_model(ALAE_CONFIG, training_artifacts_dir=ALAE_ARTIFACTS, device=device)
        decode_to_folder(alae_model, x_0, input_folder, args.decode_batch)
        if not args.no_fid:
            reference_folder(args.n_reference, args.data_dir, device, alae_model, args.decode_batch)


def evaluate_run(J, seed, args, device, checkpoint):
    """Transport the held-out latents, score them and complete the run's metrics.

    Args:
        J: Number of mixture components.
        seed: Random seed.
        args: Parsed CLI arguments.
        device: Device used for transport, decoding and FID.
        checkpoint: Checkpoint dict of the run.
    """
    from baselines.eval_alae_baseline import IMAGE_DIR, load_source_latents, score_transport

    # The scoring records these attributes, and it is shared with the baseline table.
    run_args = argparse.Namespace(**vars(args), model="lightsbb", seed=seed, n_steps=None)
    torch.manual_seed(seed)

    x_0 = load_source_latents(run_args)
    model, model_inv = load_models(checkpoint, device)

    start = time.perf_counter()
    x_1 = transport(model, model_inv, x_0.to(device), args.beta, args.safe_t).cpu()
    transport_seconds = time.perf_counter() - start
    if not torch.isfinite(x_1).all():
        raise ValueError("non-finite transported latents")

    del model, model_inv
    prepare_shared_images(run_args, x_0, device)

    tag = f"{sweep_name(args)}/J{J}"
    architecture = {k: v for k, v in checkpoint.items() if not isinstance(v, dict)}
    metrics_path = Path(score_transport(run_args, tag, x_0, x_1, architecture, device))

    metrics = json.loads(metrics_path.read_text())
    metrics.update(J=J, transport_seconds=transport_seconds,
                   train_seconds=checkpoint["train_seconds"],
                   peak_train_memory_gb=checkpoint["peak_train_memory_gb"],
                   n_parameters=checkpoint["n_parameters"])
    write_atomic(metrics_path, lambda tmp: Path(tmp).write_text(json.dumps(metrics, indent=2)))

    Path(f"{metrics_path.parent}.tar.gz").unlink(missing_ok=True)
    if not args.keep_images:
        output_folder = IMAGE_DIR / f"{tag}_s{seed}_n{args.n_images}"
        for image in output_folder.glob("*.png"):
            image.unlink()
        output_folder.rmdir()


def write_summary(args):
    """Rebuild the per-run summary table from every finished run."""
    folder = results_dir(args)
    rows = [json.loads(path.read_text()) for path in sorted(folder.glob("J*/seed_*/metrics.json"))]
    rows.sort(key=lambda row: (row["J"], row["seed"]))

    def write(tmp):
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, SUMMARY_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    folder.mkdir(parents=True, exist_ok=True)
    write_atomic(folder / "summary.csv", write)


def archive_path(args):
    """Return the archive of this sweep at the repo root."""
    return ROOT / f"alae_{sweep_name(args)}.tar.gz"


def write_archive(args):
    """Bundle the results and checkpoints so far into one archive at the repo root."""

    def write(tmp):
        with tarfile.open(tmp, "w:gz") as tar:
            for folder in (results_dir(args), checkpoint_dir(args)):
                if folder.is_dir():
                    tar.add(folder, arcname=folder.relative_to(ROOT).as_posix())

    write_atomic(archive_path(args), write)


def run_one(J, seed, args, device):
    """Train and evaluate one run, skipping what an earlier launch already finished.

    A failed run is recorded and the sweep moves on: an unstable J is itself a result.

    Args:
        J: Number of mixture components.
        seed: Random seed.
        args: Parsed CLI arguments.
        device: Device of this run.
    """
    run_dir = results_dir(args) / f"J{J}" / f"seed_{seed}"
    if (run_dir / "metrics.json").exists():
        print(f"J={J} seed={seed}: already complete, skipping", flush=True)
        return

    print(f"J={J} seed={seed}: starting on {device}", flush=True)
    failure_path = run_dir / "failure.json"
    checkpoint_path = checkpoint_dir(args) / f"J{J}_s{seed}.pkl"
    try:
        if checkpoint_path.exists():
            with open(checkpoint_path, "rb") as f:
                checkpoint = pickle.load(f)
        else:
            checkpoint = train_run(J, seed, args, device, checkpoint_path)
        evaluate_run(J, seed, args, device, checkpoint)
        failure_path.unlink(missing_ok=True)
    except (RuntimeError, ValueError) as error:
        print(f"J={J} seed={seed}: failed with {error}", flush=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(json.dumps(
            {"J": J, "seed": seed, "error": str(error), "traceback": traceback.format_exc()},
            indent=2))
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_summary(args)
    write_archive(args)


def main():
    """Run every requested (J, seed) pair in turn on the chosen device."""
    args = parse_args()
    if args.no_grad_clip:
        # The training loops clip unconditionally and look the function up at each step.
        torch.nn.utils.clip_grad_norm_ = lambda *_, **__: None
    torch.set_num_threads(args.threads)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    for J in args.js:
        for seed in args.seeds:
            run_one(J, seed, args, device)
    print(f"archive: {archive_path(args)}", flush=True)


if __name__ == "__main__":
    main()
