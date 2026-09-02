"""Decode baseline outputs on held-out ALAE latents and score age and identity."""

import argparse
import json
import os
import pickle
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

LIGHT_SBB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIGHT_SBB))
# ALAE's own modules import each other flatly ("from model import Model"), so its
# package directory has to sit on the path as well.
sys.path.insert(0, str(LIGHT_SBB / "alae"))

from alae.alae_ffhq_inference import decode, load_model  # noqa: E402
from baselines import BASELINES  # noqa: E402
from baselines.alae_data import DIM, MODEL_DIR, ROOT, load_alae_splits  # noqa: E402
from metrics_alae import compute_average_age, compute_cosine_similarity  # noqa: E402

RESULTS_DIR = ROOT / "results" / "alae_baselines"
IMAGE_DIR = ROOT / "artifacts" / "alae_baseline_images"
ALAE_CONFIG = "alae/ffhq.yaml"
ALAE_ARTIFACTS = "alae/training_artifacts/ffhq/"


def decode_to_folder(alae_model, latents, folder, batch_size=16):
    """Decode latents to PNG files named by their index in ``latents``.

    Args:
        alae_model: Loaded ALAE model.
        latents: (n, DIM) tensor of latent codes on CPU.
        folder: Destination directory, created if missing.
        batch_size: Latents decoded per call.
    """
    os.makedirs(folder, exist_ok=True)

    # Decoding dominates the run, so a folder left by an earlier eval is reused
    # rather than rewritten. The shared input folder is the usual case.
    if len(list(Path(folder).glob("*.png"))) == len(latents):
        return

    with torch.no_grad():
        for start in tqdm(range(0, len(latents), batch_size)):
            batch = latents[start:start + batch_size]
            decoded = decode(alae_model, batch)
            decoded = ((decoded * 0.5 + 0.5) * 255).clamp(0, 255)
            decoded = decoded.type(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

            for offset, img in enumerate(decoded):
                Image.fromarray(img).save(os.path.join(folder, f"{start + offset:05d}.png"))


def save_results(run_dir, args, ages, similarities, n_transported, checkpoint,
                 fid=None, fid_n_images=None):
    """Write per-image scores and the summary metrics, then archive the run.

    Args:
        run_dir: Destination directory for this model/seed.
        args: Parsed CLI arguments, recorded for provenance.
        ages: Array of InsightFace age estimates.
        similarities: Dict mapping image name to cosine similarity.
        n_transported: Number of held-out latents transported.
        checkpoint: Loaded checkpoint, recorded so the architecture is traceable.
        fid: FID against the real child reference, omitted when not computed.
        fid_n_images: Number of generated images the FID was computed on.

    Returns:
        Path to the written ``metrics.json``.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    sims = np.array(list(similarities.values()))
    metrics = {
        "model": args.model,
        "seed": args.seed,
        "n_transported": n_transported,
        "n_faces_detected": len(ages),
        "avg_age": float(np.mean(ages)),
        "std_age": float(np.std(ages)),
        "median_age": float(np.median(ages)),
        "pct_age_leq_18": float(100.0 * (ages <= 18).mean()),
        "identity_sim_mean": float(sims.mean()),
        "identity_sim_std": float(sims.std()),
        "n_steps": args.n_steps,
        "architecture": {k: v for k, v in checkpoint.items()
                         if not isinstance(v, dict)},
    }

    if fid is not None:
        metrics["fid"] = float(fid)
        metrics["fid_n_images"] = fid_n_images
        metrics["fid_n_reference"] = args.n_reference

    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    np.save(run_dir / "ages.npy", ages)
    np.savez(run_dir / "identity_sim.npz",
             names=np.array(list(similarities.keys())), values=sims)

    archive = f"{run_dir}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(run_dir, arcname=run_dir.name)
    print(archive)

    return metrics_path


def main():
    """Transport held-out adult latents, decode them, and record the metrics."""
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(BASELINES), default="otcfm")
    p.add_argument("--n-images", type=int, default=1000)
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--decode-batch", type=int, default=8,
                   help="latents decoded per ALAE call; lower it if memory is tight")
    p.add_argument("--threads", type=int, default=8,
                   help="CPU threads torch, InsightFace and OpenCV may use")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--no-fid", action="store_true",
                   help="skip the FID pass, which decodes the reference set once")
    p.add_argument("--n-reference", type=int, default=1000,
                   help="real child faces the FID is measured against")
    p.add_argument("--device", default="cuda:3")
    args = p.parse_args()

    # Imported here because fid_alae imports this module, so a module-level import
    # would be circular.
    from baselines.fid_alae import score_folder

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(device)

    splits = load_alae_splits(args.data_dir)
    x_inds_test = splits["x_inds_test"]
    if args.n_images > len(x_inds_test):
        raise ValueError(f"only {len(x_inds_test)} held-out source latents available")

    inds = x_inds_test[:args.n_images]
    x_0 = torch.tensor(splits["test_latents"][inds])
    print(f"transporting {len(x_0)} held-out source latents")

    with open(MODEL_DIR / f"{args.model}_s{args.seed}.pkl", "rb") as f:
        checkpoint = pickle.load(f)

    model = BASELINES[args.model].from_checkpoint(checkpoint, input_dim=DIM, device=device)
    x_1 = model.transport(x_0, n_steps=args.n_steps).cpu()

    alae_model = load_model(ALAE_CONFIG, training_artifacts_dir=ALAE_ARTIFACTS,
                            device=device)

    input_folder = IMAGE_DIR / f"input_n{args.n_images}"
    output_folder = IMAGE_DIR / f"{args.model}_s{args.seed}_n{args.n_images}"
    decode_to_folder(alae_model, x_0, input_folder, args.decode_batch)
    decode_to_folder(alae_model, x_1, output_folder, args.decode_batch)

    ages = np.asarray(compute_average_age(output_folder, n_threads=args.threads))
    if len(ages) == 0:
        raise RuntimeError("no faces detected in any transported image")
    print(f"Age <= 18 (%): {100.0 * (ages <= 18).mean():.1f}")

    similarities = compute_cosine_similarity(input_folder, output_folder)

    fid = fid_n_images = None
    if not args.no_fid:
        # Scored before the results are archived, so the tar carries the FID too.
        fid, fid_n_images = score_folder(
            output_folder, args.n_images, args.n_reference, args.data_dir, device,
            alae_model=alae_model, decode_batch=args.decode_batch,
        )
        print(f"FID: {fid:.2f}  ({fid_n_images} generated vs "
              f"{args.n_reference} reference)")

    run_dir = RESULTS_DIR / args.model / f"seed_{args.seed}"
    print(save_results(run_dir, args, ages, similarities, len(x_0), checkpoint,
                       fid, fid_n_images))


if __name__ == "__main__":
    main()
