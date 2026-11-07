"""Score saved ALAE image folders: age, identity with a face-recognition model, FID with uncertainty.

Needs no model or decoder, only the PNG folders. Run from light_sbb/:

    python baselines/score_alae_folders.py --device cuda:0
"""

import argparse
import csv
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import linalg

LIGHT_SBB = Path(__file__).resolve().parents[1]
ROOT = LIGHT_SBB.parent
RESULTS_DIR = ROOT / "results" / "alae_table3_folders"
ARCHIVE = ROOT / "alae_table3_folders.tar.gz"

# Paired folders: input adults and the outputs transported from them, in the same order.
PAIRED = {"input": "input_child_test_sim", "sb": "sb_child_test_sim_e0.1", "sbb": "sbb_child_test_sim_e0.1"}
# Generated children scored against real ones.
UNPAIRED = {"real": "real_child_test", "sb": "sb_child_test", "sbb": "sbb_child_test"}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
CHILD_AGE = 18
N_BOOT = 1000
N_SHOW = 8


def parse_args():
    """Parse the folder location, the device and the resampling settings."""
    p = argparse.ArgumentParser()
    p.add_argument("--image-root", default=str(LIGHT_SBB / "alae_images"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n-fid-draws", type=int, default=50, help="half-size subsets drawn to measure the FID spread")
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def list_images(folder):
    """Return the sorted image file names of ``folder``."""
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS))


def bootstrap(values, rng, statistic=np.mean):
    """Return the statistic and its 95% bootstrap interval over images.

    Args:
        values: (n,) per-image values, NaN-free.
        rng: Numpy generator.
        statistic: Function of a 1D array.

    Returns:
        Dict with ``value``, ``ci_low``, ``ci_high``, ``sd_over_images`` and ``n``.
    """
    values = np.asarray(values, dtype=float)
    draws = [statistic(values[rng.integers(0, len(values), len(values))]) for _ in range(N_BOOT)]
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"value": float(statistic(values)), "ci_low": float(low), "ci_high": float(high),
            "sd_over_images": float(values.std(ddof=1)), "n": int(len(values))}


def face_records(app, folder, names):
    """Run InsightFace on every image and keep the age of all faces and the largest face's embedding.

    Args:
        app: Prepared ``FaceAnalysis``.
        folder: Image directory.
        names: File names to read.

    Returns:
        Dict mapping each name to ``{"ages": [...], "embedding": array or None}``.
    """
    records = {}
    for name in names:
        img = cv2.imread(os.path.join(folder, name))
        faces = app.get(img) if img is not None else []
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), default=None)
        records[name] = {"ages": [float(f.age) for f in faces],
                         "largest_age": float(largest.age) if largest is not None else None,
                         "embedding": None if largest is None else np.asarray(largest.normed_embedding)}
    return records


def age_summary(records, rng):
    """Summarize ages both as the paper did (every detected face) and per image (largest face)."""
    all_faces = np.array([a for r in records.values() for a in r["ages"]])
    largest = np.array([r["largest_age"] for r in records.values() if r["largest_age"] is not None])
    out = {"n_images": len(records), "n_images_with_face": int(len(largest)), "n_faces": int(len(all_faces))}
    if len(all_faces):
        out["all_faces_mean_age"] = bootstrap(all_faces, rng)
        out["all_faces_pct_child"] = bootstrap(100.0 * (all_faces <= CHILD_AGE), rng)
    if len(largest):
        out["per_image_mean_age"] = bootstrap(largest, rng)
        out["per_image_pct_child"] = bootstrap(100.0 * (largest <= CHILD_AGE), rng)
    return out


def pixel_cosine(path_a, path_b):
    """Cosine similarity of two flattened RGB images, the paper's identity score."""
    a = np.asarray(Image.open(path_a).convert("RGB"), dtype=np.float32).ravel()
    b = np.asarray(Image.open(path_b).convert("RGB"), dtype=np.float32).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def pair_names(input_names, output_names):
    """Pair input and output files by name when they match, else by sorted position.

    Returns:
        Tuple of the list of (input, output) name pairs and the pairing mode.
    """
    if set(input_names) == set(output_names):
        return [(n, n) for n in input_names], "by_name"
    if len(input_names) != len(output_names):
        raise ValueError(f"cannot pair {len(input_names)} inputs with {len(output_names)} outputs")
    return list(zip(input_names, output_names)), "by_sorted_position"


def inception_features(folder, names, device, batch_size):
    """Return the 2048-d pool features pytorch-fid uses, one row per image."""
    from pytorch_fid.inception import InceptionV3

    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    feats = []
    for start in range(0, len(names), batch_size):
        batch = [np.asarray(Image.open(os.path.join(folder, n)).convert("RGB"), dtype=np.float32) / 255.0
                 for n in names[start:start + batch_size]]
        x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            feats.append(model(x)[0].squeeze(-1).squeeze(-1).cpu().numpy())
    return np.concatenate(feats).astype(np.float64)


def frechet(feat_a, feat_b):
    """Frechet distance between Gaussian fits of two feature sets."""
    mu_a, mu_b = feat_a.mean(0), feat_b.mean(0)
    cov_a, cov_b = np.cov(feat_a, rowvar=False), np.cov(feat_b, rowvar=False)
    covmean, _ = linalg.sqrtm(cov_a @ cov_b, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(len(cov_a)) * 1e-6
        covmean = linalg.sqrtm((cov_a + offset) @ (cov_b + offset))
    covmean = covmean.real
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(cov_a) + np.trace(cov_b) - 2 * np.trace(covmean))


def fid_with_uncertainty(feat_gen, feat_real, rng, n_draws):
    """Return the FID at equal sample size, and its spread over half-size subsets.

    Resampling with replacement duplicates images and inflates the FID, so the spread is taken
    over subsets drawn without replacement, at half the size, on both sides.
    """
    n = min(len(feat_gen), len(feat_real))
    gen, real = feat_gen[:n], feat_real[:n]
    half = n // 2
    draws = [frechet(gen[rng.permutation(n)[:half]], real[rng.permutation(n)[:half]]) for _ in range(n_draws)]
    return {"value": frechet(gen, real), "n_generated": n, "n_real": n, "half_size": half,
            "half_size_mean": float(np.mean(draws)), "half_size_sd": float(np.std(draws, ddof=1)),
            "half_size_low": float(np.percentile(draws, 2.5)), "half_size_high": float(np.percentile(draws, 97.5))}


def copy_examples(folder, names, dest):
    """Copy a few images into the archive so they can be inspected."""
    os.makedirs(dest, exist_ok=True)
    for name in names[:N_SHOW]:
        shutil.copy(os.path.join(folder, name), os.path.join(dest, name))


def main():
    """Score both folder families and write one summary and one archive."""
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    root = Path(args.image_root)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0)

    summary = {"image_root": str(root), "child_age": CHILD_AGE, "folders": {}}
    paired_records = {}
    for key, sub in {**{f"paired_{k}": v for k, v in PAIRED.items()},
                     **{f"unpaired_{k}": v for k, v in UNPAIRED.items()}}.items():
        folder = root / sub
        names = list_images(folder)
        records = face_records(app, folder, names)
        first = Image.open(folder / names[0])
        summary["folders"][key] = {"folder": sub, "n_files": len(names), "first_files": names[:5],
                                   "image_size": list(first.size), **age_summary(records, rng)}
        if key.startswith("paired_"):
            paired_records[key.removeprefix("paired_")] = (names, records)
        with open(RESULTS_DIR / f"ages_{key}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "largest_face_age", "all_face_ages"])
            for name, r in records.items():
                writer.writerow([name, r["largest_age"], ";".join(f"{a:.1f}" for a in r["ages"])])
        copy_examples(folder, names, RESULTS_DIR / "examples" / key)
        print(key, json.dumps(summary["folders"][key])[:400], flush=True)

    input_names, input_records = paired_records["input"]
    for method in ("sb", "sbb"):
        names, records = paired_records[method]
        pairs, mode = pair_names(input_names, names)
        pixel = [pixel_cosine(root / PAIRED["input"] / a, root / PAIRED[method] / b) for a, b in pairs]
        arc = [float(input_records[a]["embedding"] @ records[b]["embedding"]) for a, b in pairs
               if input_records[a]["embedding"] is not None and records[b]["embedding"] is not None]
        summary[f"identity_{method}"] = {"pairing": mode, "pixel_cosine": bootstrap(pixel, rng),
                                         "arcface_cosine": bootstrap(arc, rng) if arc else None}
        print(f"identity {method}", json.dumps(summary[f"identity_{method}"]), flush=True)

    real_names = list_images(root / UNPAIRED["real"])
    real_feat = inception_features(root / UNPAIRED["real"], real_names, device, args.batch_size)
    for method in ("sb", "sbb"):
        for family, folders in (("unpaired", UNPAIRED), ("paired", PAIRED)):
            names = list_images(root / folders[method])
            feat = inception_features(root / folders[method], names, device, args.batch_size)
            summary[f"fid_{family}_{method}"] = fid_with_uncertainty(feat, real_feat, rng, args.n_fid_draws)
            print(f"fid {family} {method}", json.dumps(summary[f"fid_{family}_{method}"]), flush=True)

    summary["note"] = ("Each folder comes from one trained model, so the intervals are over images and cover "
                       "variation across faces, not across training seeds.")
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    with tarfile.open(ARCHIVE, "w:gz") as tar:
        tar.add(RESULTS_DIR, arcname=RESULTS_DIR.relative_to(ROOT).as_posix())
    print(f"archive: {ARCHIVE}", flush=True)


if __name__ == "__main__":
    main()
