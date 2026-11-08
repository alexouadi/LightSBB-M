"""Score a saved ALAE LightSBB-M checkpoint with its inverse network and with the exact inverse.

Only the pull-back of the source at ``t = 0`` differs between the two runs, and both share the
same bridge noise. Run from light_sbb/, on the state dicts of ``run_alae.py``:

    python baselines/compare_alae_inverse.py --model PATH --model-inv PATH --beta 1 --device cuda:0
"""

import argparse
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

LIGHT_SBB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIGHT_SBB))
sys.path.insert(0, str(LIGHT_SBB / "alae"))

from baselines.alae_data import ROOT, load_alae_splits  # noqa: E402
from baselines.diagnose_alae_invertibility import load_models, summarize  # noqa: E402
from run_alae_admissible import (REFERENCE_SCORES, age_scores, latent_marginal,  # noqa: E402
                                 reference_scores)
from run_alae_j_sweep import prepare_shared_images  # noqa: E402
from train_lightsbb_admissible import bass_map, invert_bass_map  # noqa: E402

RESULTS = ROOT / "results" / "alae_inverse_test"
ARCHIVE = ROOT / "alae_inverse_test.tar.gz"
TAG_ROOT = "inverse_test"
V2_FOLDER = "admissible_v2/shift_shared_s0_s0"
TILE = 256
HEADER = 24


def parse_args():
    """Parse the checkpoint paths and the evaluation settings."""
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="pickled state dict of the bridge")
    p.add_argument("--model-inv", required=True, help="pickled state dict of the inverse network")
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--safe-t", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-images", type=int, default=1000)
    p.add_argument("--n-reference", type=int, default=1000)
    p.add_argument("--n-rows", type=int, default=16, help="identities shown on the contact sheet")
    p.add_argument("--decode-batch", type=int, default=32)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--age-model-dir", default="vit-age-classifier")
    p.add_argument("--no-fid", action="store_true")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


@torch.no_grad()
def transport_from(model, y_0, beta, safe_t, seed):
    """Run the bridge from ``y_0`` and push the endpoint back by the Bass map.

    Args:
        model: float64 ``LightSBM``.
        y_0: (n, d) float64 sources in Y-space.
        beta: Bass parameter.
        safe_t: Margin of the Bass map at the terminal time.
        seed: Seed of the bridge noise, shared between the compared inverses.

    Returns:
        (n, d) float32 transported latents on CPU.
    """
    torch.manual_seed(seed)
    return bass_map(model, model(y_0), 1.0 - safe_t, beta).float().cpu()


def compare_transports(model, model_inv, x_0, args, device):
    """Transport the sources with both inverses and measure how far the inverses are apart.

    Args:
        model: float64 ``LightSBM``.
        model_inv: Inverse network.
        x_0: (n, d) float32 source latents on CPU.
        args: Parsed arguments.
        device: Device of the models.

    Returns:
        Tuple of the dict of transported latents per variant and a JSON-ready inverse report.
    """
    x = x_0.to(device)
    with torch.no_grad():
        y_net = model_inv(torch.zeros((len(x), 1), device=device), x).double()
    y_exact, residual = invert_bass_map(model, x.double(), 0.0, args.beta)
    x_norm = x.double().norm(dim=1)
    report = {
        "exact_residual_relative_to_x": summarize((residual / x_norm).cpu().numpy()),
        "net_residual_relative_to_x": summarize(
            ((bass_map(model, y_net, 0.0, args.beta) - x.double()).norm(dim=1) / x_norm).cpu().numpy()),
        "net_to_exact_relative_to_x": summarize(((y_net - y_exact).norm(dim=1) / x_norm).cpu().numpy()),
        "exact_displacement_relative_to_x": summarize(((y_exact - x.double()).norm(dim=1) / x_norm).cpu().numpy()),
    }
    transports = {name: transport_from(model, y, args.beta, args.safe_t, args.seed)
                  for name, y in (("old_net", y_net), ("old_exact", y_exact))}
    return transports, report


def save_contact_sheet(columns, n_rows, path):
    """Save one row per identity, with one labelled column per folder.

    Args:
        columns: List of ``(label, folder)``; files are paired by name.
        n_rows: Number of identities.
        path: Destination PNG.
    """
    names = [p.name for p in sorted(Path(columns[0][1]).glob("*.png"))[:n_rows]]
    sheet = Image.new("RGB", (TILE * len(columns), HEADER + TILE * len(names)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, folder) in enumerate(columns):
        draw.text((i * TILE + 4, 6), label, fill="black")
        for j, name in enumerate(names):
            tile = Image.open(Path(folder) / name).convert("RGB").resize((TILE, TILE))
            sheet.paste(tile, (i * TILE, HEADER + j * TILE))
    sheet.save(path)


def main():
    """Score both inverses on the same checkpoint and archive the comparison."""
    args = parse_args()
    from insightface.app import FaceAnalysis

    from baselines.eval_alae_baseline import IMAGE_DIR, load_source_latents, score_transport
    from baselines.score_alae_folders import load_age_classifier

    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    np.random.seed(args.seed)
    model, model_inv = load_models(args, device)
    run_args = argparse.Namespace(**{**vars(args), "model": "lightsbb", "n_steps": None})

    splits = load_alae_splits(args.data_dir)
    x_0 = load_source_latents(run_args)
    children = torch.as_tensor(splits["test_latents"][splits["y_inds_test"]])
    transports, inverse_report = compare_transports(model, model_inv, x_0, args, device)
    results = {"epsilon": float(model.epsilon), "beta": args.beta, "inverse": inverse_report, "variants": {}}

    prepare_shared_images(run_args, x_0, device)
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0)
    processor, classifier = load_age_classifier(args.age_model_dir, device)

    def score(folder):
        return age_scores(folder, app, processor, classifier, args.n_images, device)

    columns = [("input", IMAGE_DIR / f"input_n{args.n_images}")]
    for name, x_1 in transports.items():
        if not torch.isfinite(x_1).all():
            raise ValueError(f"non-finite transported latents in {name}")
        tag = f"{TAG_ROOT}/{name}"
        metrics_path = Path(score_transport(run_args, tag, x_0, x_1, {"beta": args.beta}, device))
        folder = IMAGE_DIR / f"{tag}_s{args.seed}_n{args.n_images}"
        results["variants"][name] = {"metrics": json.loads(metrics_path.read_text()),
                                     "ages": score(folder),
                                     "latent_marginal": latent_marginal(x_0, x_1, children)}
        columns.append((name, folder))

    results["ages_real_children"] = reference_scores(score, IMAGE_DIR / f"reference_children_n{args.n_reference}")
    v2_folder = IMAGE_DIR / f"{V2_FOLDER}_n{args.n_images}"
    if v2_folder.is_dir():
        columns.append(("v2 shift_shared", v2_folder))

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "results.json").write_text(json.dumps(results, indent=2, default=str))
    save_contact_sheet(columns, args.n_rows, RESULTS / "contact_sheet.png")
    with tarfile.open(ARCHIVE, "w:gz") as tar:
        tar.add(RESULTS, arcname=RESULTS.relative_to(ROOT).as_posix())
    print(f"finished; real children cached at {REFERENCE_SCORES}; wrote {ARCHIVE}", flush=True)


if __name__ == "__main__":
    main()
