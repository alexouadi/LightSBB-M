"""Score decoded ALAE outputs with FID against real held-out child faces.

Runs on the PNG folders left behind by ``eval_alae_baseline.py``, so adding FID to a
finished run costs one decode of the reference set and no retraining.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import torch
from pytorch_fid.fid_score import calculate_fid_given_paths

LIGHT_SBB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIGHT_SBB))
sys.path.insert(0, str(LIGHT_SBB / "alae"))

from alae.alae_ffhq_inference import load_model  # noqa: E402
from baselines.alae_data import load_alae_splits  # noqa: E402
from baselines.eval_alae_baseline import (ALAE_ARTIFACTS, ALAE_CONFIG,  # noqa: E402
                                          IMAGE_DIR, decode_to_folder)

DIMS = 2048


@contextmanager
def capped(folder, n_images):
    """Yield ``folder``, or a temporary copy holding only its first ``n_images``.

    FID is strongly biased by sample size, so every method has to be scored on the
    same count even when more images were decoded.

    Args:
        folder: Directory of decoded PNG images.
        n_images: Cap on the number of images, or None to use them all.

    Yields:
        Path to the directory to score.
    """
    names = sorted(f for f in os.listdir(folder) if f.endswith(".png"))
    if n_images is None or len(names) <= n_images:
        yield folder
        return

    tmp = tempfile.mkdtemp(prefix="fid_")
    try:
        for name in names[:n_images]:
            shutil.copy2(os.path.join(folder, name), os.path.join(tmp, name))
        yield tmp
    finally:
        shutil.rmtree(tmp)


def reference_folder(n_reference, data_dir, device):
    """Return the folder of decoded real child faces, decoding it once if missing.

    Args:
        n_reference: Number of held-out child latents to decode.
        data_dir: Directory holding the ALAE ``.npy`` files.
        device: Device the ALAE decoder runs on.

    Returns:
        Path to the reference folder.
    """
    folder = IMAGE_DIR / f"reference_children_n{n_reference}"
    if folder.is_dir() and len(list(folder.glob("*.png"))) == n_reference:
        return folder

    splits = load_alae_splits(data_dir)
    y_inds_test = splits["y_inds_test"]
    if n_reference > len(y_inds_test):
        raise ValueError(f"only {len(y_inds_test)} held-out child latents available")

    latents = torch.tensor(splits["test_latents"][y_inds_test[:n_reference]])
    alae_model = load_model(ALAE_CONFIG, training_artifacts_dir=ALAE_ARTIFACTS,
                            device=device)
    decode_to_folder(alae_model, latents, folder)
    return folder


def main():
    """Compute FID for one decoded folder and merge it into its metrics.json."""
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True,
                   help="folder of decoded PNGs to score")
    p.add_argument("--metrics",
                   help="metrics.json to merge the score into; skipped if omitted")
    p.add_argument("--n-images", type=int, default=1000,
                   help="cap on generated images scored, so every method matches")
    p.add_argument("--n-reference", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--threads", type=int, default=8,
                   help="CPU threads torch may use")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    torch.set_num_threads(args.threads)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(device)

    ref_folder = reference_folder(args.n_reference, args.data_dir, device)

    with capped(args.images, args.n_images) as scored:
        n_scored = len([f for f in os.listdir(scored) if f.endswith(".png")])
        fid = calculate_fid_given_paths([str(scored), str(ref_folder)],
                                        batch_size=args.batch_size, device=device,
                                        dims=DIMS)

    print(f"FID: {fid:.2f}  ({n_scored} generated vs {args.n_reference} reference)")

    if args.metrics:
        path = Path(args.metrics)
        with open(path) as f:
            metrics = json.load(f)
        metrics["fid"] = float(fid)
        metrics["fid_n_images"] = n_scored
        metrics["fid_n_reference"] = args.n_reference
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(path)


if __name__ == "__main__":
    main()
