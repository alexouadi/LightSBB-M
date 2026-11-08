"""Train, diagnose and score the admissible LightSBB variants on the ALAE latents.

Run from light_sbb/, one process per device. Each finished run refreshes the archive:

    python run_alae_admissible.py --configs newton sb --seeds 0 1 --device cuda:0
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
from PIL import Image

LIGHT_SBB = Path(__file__).resolve().parent
sys.path.insert(0, str(LIGHT_SBB))
sys.path.insert(0, str(LIGHT_SBB / "alae"))

from baselines.alae_data import DIM, MODEL_DIR, ROOT, load_alae_splits  # noqa: E402
from baselines.diagnose_alae_invertibility import (curvature_along_path,  # noqa: E402
                                                   summarize, weight_diagnostics)
from lightsbm import LightSBM  # noqa: E402
from run_alae_j_sweep import prepare_shared_images, shared_lock, write_atomic  # noqa: E402
from train_lightsbb_admissible import (bass_map, curvature_floor, invert_bass_map,  # noqa: E402
                                       mixture_usage, training_sbb_admissible)

RESULTS = ROOT / "results" / "alae_admissible_v2"
CHECKPOINTS = MODEL_DIR / "admissible_v2"
ARCHIVE = ROOT / "alae_admissible_v2.tar.gz"
LOCK = ROOT / "artifacts" / ".alae_admissible_v2.lock"
REFERENCE_SCORES = RESULTS / "real_children_scores.json"
N_PROJECTIONS = 2000
AGE_BATCH = 32
N_GRID = 8
GRID_SIZE = 256

# r_init: "mixed" draws J/2 sources and J/2 targets, "target" draws targets, "target_shift" also
# removes S * mean(source), so that the conditional mean r + S x starts on a target at the source mean.
CONFIGS = {
    "newton": {"r_init": "mixed", "s_init": 1.0, "shared_s": False},
    "shift": {"r_init": "target_shift", "s_init": 1.0, "shared_s": False},
    "shift_shared": {"r_init": "target_shift", "s_init": 1.0, "shared_s": True},
    "shift_shared_low": {"r_init": "target_shift", "s_init": 0.6, "shared_s": True},
    "sb": {"r_init": "target", "s_init": 0.1, "shared_s": False},
}


def parse_args():
    """Parse the training and evaluation settings."""
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", choices=sorted(CONFIGS), default=["shift_shared"])
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
    p.add_argument("--age-model-dir", default="vit-age-classifier")
    p.add_argument("--no-fid", action="store_true")
    return p.parse_args()


def initial_centers(splits, config, device, J):
    """Draw the initial mixture centers.

    Args:
        splits: Output of ``load_alae_splits``.
        config: Entry of ``CONFIGS``.
        device: Target device.
        J: Number of components.

    Returns:
        (J, DIM) tensor of centers.
    """
    x_pool = torch.as_tensor(splits["x_train"], device=device)
    y_pool = torch.as_tensor(splits["y_train"], device=device)
    targets = y_pool[torch.randperm(len(y_pool))[:J]]
    if config["r_init"] == "target":
        return targets
    if config["r_init"] == "target_shift":
        return targets - config["s_init"] * x_pool.mean(0)
    sources = x_pool[torch.randperm(len(x_pool))[:J // 2]]
    return torch.cat([sources, y_pool[torch.randperm(len(y_pool))[:J - J // 2]]])


def transport(model, x_0, args, is_sb):
    """Map source latents to the target, with the exact inverse and the Bass map for LightSBB.

    Args:
        model: Trained ``LightSBM``.
        x_0: (n, DIM) source latents on the model's device.
        args: Parsed arguments.
        is_sb: Whether the model is the plain LightSB-M reference.

    Returns:
        (n, DIM) transported float32 latents on the model's device.
    """
    if is_sb:
        return model(x_0)
    y_0, _ = invert_bass_map(model, x_0, 0.0, args.beta)
    y_T = model(y_0)
    x_1 = bass_map(model.double(), y_T.double(), 1.0 - args.safe_t, args.beta).float()
    model.float()
    return x_1


def diagnostics(model, splits, args, device):
    """Measure the curvature, the mixture usage and the solver residuals on held-out latents.

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
    y_T = model(y_0)
    usage = mixture_usage(model, y_0, curvature_floor(args.beta, args.margin))

    precise = model.double()
    results = {
        "epsilon": float(precise.epsilon),
        "usage": usage,
        "weights": weight_diagnostics(precise, y_0.double(), args.beta),
        "inverse_residual_t0": summarize((residual_0 / adults.norm(dim=1)).cpu().numpy()),
        "inverse_residual_terminal": summarize((residual_T / children.norm(dim=1)).cpu().numpy()),
        "curvature": curvature_along_path(precise, y_0[:args.n_points].double(),
                                          y_T[:args.n_points].double(), args),
    }
    model.float()
    return results


def sliced_w2(a, b, n_projections=N_PROJECTIONS):
    """Return the sliced 2-Wasserstein distance between two equal-size samples."""
    directions = torch.randn(n_projections, a.shape[1])
    directions /= directions.norm(dim=1, keepdim=True)
    gaps = (a @ directions.T).sort(0).values - (b @ directions.T).sort(0).values
    return float(gaps.pow(2).mean().sqrt())


def latent_marginal(x_0, x_1, children):
    """Compare transported latents with held-out children, with adults and a split of children as anchors.

    Args:
        x_0: (n, DIM) source latents on CPU.
        x_1: (n, DIM) transported latents on CPU.
        children: (m, DIM) held-out child latents on CPU.

    Returns:
        JSON-ready dict; ``sliced_w2_split_of_children`` is the sampling floor.
    """
    n = min(len(x_1), len(children) // 2)
    reference = children[:n]
    variance_ratio = x_1.var(0) / children.var(0)
    return {
        "n": n,
        "mean_gap": float((x_1.mean(0) - children.mean(0)).norm()),
        "mean_gap_adults": float((x_0.mean(0) - children.mean(0)).norm()),
        "variance_ratio": summarize(variance_ratio.numpy()),
        "sliced_w2": sliced_w2(x_1[:n], reference),
        "sliced_w2_adults": sliced_w2(x_0[:n], reference),
        "sliced_w2_split_of_children": sliced_w2(children[n:2 * n], reference),
    }


def age_scores(folder, app, processor, classifier, n_images, device):
    """Score a folder with bordered InsightFace and the ViT age classifier.

    Args:
        folder: Image directory.
        app: Prepared ``FaceAnalysis``.
        processor: Image processor of the classifier.
        classifier: Age classifier on ``device``.
        n_images: Number of images to score.
        device: Device of the classifier.

    Returns:
        JSON-ready dict.
    """
    from baselines.score_alae_folders import (AGE_PAD, age_summary, classifier_ages,
                                              classifier_summary, face_records, list_images)

    rng = np.random.default_rng(0)
    names = list_images(folder)[:n_images]
    labels = [classifier.config.id2label[i] for i in range(classifier.config.num_labels)]
    probs = classifier_ages(processor, classifier, folder, names, device, AGE_BATCH)
    return {"insightface_border": age_summary(face_records(app, folder, names, AGE_PAD), rng),
            "classifier": classifier_summary(probs, labels, rng)}


def reference_scores(score, folder):
    """Return the real-children age scores, computing them once across processes."""
    with shared_lock(LOCK):
        if not REFERENCE_SCORES.exists():
            REFERENCE_SCORES.parent.mkdir(parents=True, exist_ok=True)
            REFERENCE_SCORES.write_text(json.dumps(score(folder), indent=2))
    return json.loads(REFERENCE_SCORES.read_text())


def save_grid(folder, path):
    """Save the first images of a folder as one small strip, to inspect the outputs by eye."""
    names = sorted(Path(folder).glob("*.png"))[:N_GRID]
    strip = Image.new("RGB", (GRID_SIZE * len(names), GRID_SIZE))
    for i, name in enumerate(names):
        strip.paste(Image.open(name).convert("RGB").resize((GRID_SIZE, GRID_SIZE)), (i * GRID_SIZE, 0))
    strip.save(path)


def write_archive():
    """Bundle every finished run into one archive at the repo root."""

    def write(tmp):
        with tarfile.open(tmp, "w:gz") as tar:
            for folder in (RESULTS, CHECKPOINTS):
                if folder.is_dir():
                    tar.add(folder, arcname=folder.relative_to(ROOT).as_posix())

    write_atomic(ARCHIVE, write)


def run_one(config_name, seed, args, device):
    """Train, diagnose and score one configuration and seed.

    Args:
        config_name: Key of ``CONFIGS``.
        seed: Random seed.
        args: Parsed arguments.
        device: Training device.
    """
    from insightface.app import FaceAnalysis

    from baselines.eval_alae_baseline import IMAGE_DIR, load_source_latents, score_transport
    from baselines.score_alae_folders import load_age_classifier

    config = CONFIGS[config_name]
    is_sb = config_name == "sb"
    name = f"{config_name}_s{seed}"
    torch.manual_seed(seed)
    np.random.seed(seed)
    splits = load_alae_splits(args.data_dir)
    x_pool = torch.as_tensor(splits["x_train"], device=device)
    y_pool = torch.as_tensor(splits["y_train"], device=device)

    model = LightSBM(dim=DIM, n_potentials=args.J, epsilon=args.eps, S_diagonal_init=config["s_init"],
                     is_diagonal=True, sampling_batch_size=256).to(device)
    model.init_r_by_samples(initial_centers(splits, config, device, args.J))

    start = time.perf_counter()
    model, log = training_sbb_admissible(
        x_pool, y_pool, model, args.beta, K=1 if is_sb else args.k, lr=args.lr, n_epochs=args.n_epochs,
        min_epoch=args.min_epoch, batch_size=args.batch_size, eps=args.eps, safe_t=args.safe_t,
        damping=args.damping, margin=None if is_sb else args.margin, shared_s=config["shared_s"],
        print_every=args.print_every, device=device)
    results = {"config": config, "train_seconds": time.perf_counter() - start, "stage_log": log}
    if not is_sb:
        results["diagnostics"] = diagnostics(model, splits, args, device)

    run_args = argparse.Namespace(**vars(args), model="lightsbb", seed=seed, n_steps=None)
    torch.manual_seed(seed)
    x_0 = load_source_latents(run_args)
    x_1 = transport(model, x_0.to(device), args, is_sb).cpu()
    if not torch.isfinite(x_1).all():
        raise ValueError("non-finite transported latents")
    children = torch.as_tensor(splits["test_latents"][splits["y_inds_test"]])
    results["latent_marginal"] = latent_marginal(x_0, x_1, children)

    prepare_shared_images(run_args, x_0, device)
    tag = f"admissible_v2/{name}"
    metrics_path = Path(score_transport(run_args, tag, x_0, x_1, {"beta": args.beta, "J": args.J}, device))
    results["metrics"] = json.loads(metrics_path.read_text())

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0)
    processor, classifier = load_age_classifier(args.age_model_dir, device)

    def score(folder):
        return age_scores(folder, app, processor, classifier, args.n_images, device)

    output_folder = IMAGE_DIR / f"{tag}_s{seed}_n{args.n_images}"
    results["ages"] = score(output_folder)
    results["ages_real_children"] = reference_scores(score, IMAGE_DIR / f"reference_children_n{args.n_reference}")

    out_dir = RESULTS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_grid(output_folder, out_dir / "samples.png")
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINTS / f"{name}.pkl", "wb") as f:
        pickle.dump({"state_dict": state, "config": config, "seed": seed, "beta": args.beta, "eps": args.eps}, f)
    print(f"finished {name}: {results['metrics']}", flush=True)


def main():
    """Run every requested configuration and seed, refreshing the archive after each one."""
    args = parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    for config_name in args.configs:
        for seed in args.seeds:
            run_one(config_name, seed, args, device)
            with shared_lock(LOCK):
                write_archive()


if __name__ == "__main__":
    main()
