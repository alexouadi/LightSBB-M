import argparse
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from light_sbb.alae.alae_ffhq_inference import load_model, decode
from light_sbb.lightsbm import LightSBM, MLP_network


def split_data(latents: np.ndarray, age: np.ndarray, train_size: int = 60000, test_size: int = 10000):
    """Replicate run_alae.py split and return adult/child test indices."""
    test_latents = latents[train_size:]
    test_age = age[train_size:]

    x_inds_test = np.arange(test_size)[
        (test_age >= 18).reshape(-1) * (test_age != -1).reshape(-1)
    ]
    y_inds_test = np.arange(test_size)[
        (test_age < 18).reshape(-1) * (test_age != -1).reshape(-1)
    ]
    return test_latents, x_inds_test, y_inds_test


def load_transport_models(beta: float, eps: float, dim: int, n_potentials: int, s_init: float, device: torch.device):
    """Load LightSBB transport model and optional inverse model from disk."""
    model_path = Path(f"model/b{beta}_e{eps}.pkl")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model weights: {model_path}")

    model = LightSBM(
        dim=dim,
        n_potentials=n_potentials,
        epsilon=eps,
        S_diagonal_init=s_init,
        is_diagonal=True,
    ).to(device)
    with open(model_path, "rb") as f:
        model.load_state_dict(pickle.load(f), strict=True)
    model.eval()

    model_inv = None
    if beta < 100:
        inv_path = Path(f"model_inv/b{beta}_e{eps}.pkl")
        if not inv_path.exists():
            raise FileNotFoundError(f"Missing inverse model weights: {inv_path}")
        model_inv = MLP_network(input_dim=dim, t_model=32, d_model=128).to(device)
        with open(inv_path, "rb") as f:
            model_inv.load_state_dict(pickle.load(f), strict=True)
        model_inv.eval()

    return model, model_inv


@torch.no_grad()
def map_adult_latents_to_child(model, model_inv, beta: float, adult_latents: torch.Tensor, device: torch.device):
    """Map adult test latents to translated child latents using the same logic as run_alae.py."""
    adult_latents = adult_latents.to(device)
    if beta >= 100:
        y_latent_to_map = adult_latents - 1.0 / beta * model.get_drift(
            torch.zeros(len(adult_latents), device=device), adult_latents
        )
    else:
        y_latent_to_map = model_inv(torch.zeros((len(adult_latents), 1), device=device), adult_latents)

    return model(y_latent_to_map)


@torch.no_grad()
def decode_latents(alae_model, latents: torch.Tensor, batch_size: int = 64):
    """Decode latents into uint8 RGB images with shape (N, H, W, 3)."""
    decoded_batches = []
    for i in range(0, len(latents), batch_size):
        decoded = decode(alae_model, latents[i:i + batch_size].cpu())
        decoded = ((decoded * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255)
        decoded = decoded.type(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        decoded_batches.append(decoded)
    return np.concatenate(decoded_batches, axis=0)


def save_images_to_dir(images: np.ndarray, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        Image.fromarray(img).save(out_dir / f"{i:06d}.png")


@torch.no_grad()
def extract_inception_features(images: np.ndarray, batch_size: int, device: torch.device):
    """Extract 2048-D Inception features used for FID."""
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device).eval()

    feats = []
    for i in range(0, len(images), batch_size):
        batch = torch.from_numpy(images[i:i + batch_size]).float() / 255.0
        batch = batch.permute(0, 3, 1, 2).to(device)
        pred = inception(batch)[0]
        pred = pred.squeeze(-1).squeeze(-1).cpu().numpy()
        feats.append(pred)
    return np.concatenate(feats, axis=0)


def compute_balanced_accuracy(real_child_images: np.ndarray, generated_images: np.ndarray, batch_size: int,
                              device: torch.device, seed: int):
    """Train a lightweight discriminator and report balanced accuracy on held-out test split."""
    n = min(len(real_child_images), len(generated_images))
    real = real_child_images[:n]
    fake = generated_images[:n]

    x = np.concatenate([real, fake], axis=0)
    y = np.concatenate([np.ones(n, dtype=np.int64), np.zeros(n, dtype=np.int64)], axis=0)

    feats = extract_inception_features(x, batch_size=batch_size, device=device)

    x_train, x_test, y_train, y_test = train_test_split(
        feats,
        y,
        test_size=0.3,
        random_state=seed,
        stratify=y,
    )

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return balanced_accuracy_score(y_test, y_pred)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute classification and FID metrics for ALAE translation outputs.")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta value used during training.")
    parser.add_argument("--eps", type=float, default=0.1, help="Epsilon value used during training.")
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--n-potentials", type=int, default=10)
    parser.add_argument("--s-init", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Optional cap on number of test adult/child samples used for evaluation.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--alae-artifacts-dir",
        type=str,
        default="light_sbb/alae/training_artifacts/ffhq/",
        help="Directory containing ALAE inference weights/checkpoints.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    latents = np.load("data/latents.npy")
    age = np.load("data/age.npy")
    test_images = np.load("data/test_images.npy")

    test_latents, adult_inds_test, child_inds_test = split_data(latents, age)

    n_eval = min(len(adult_inds_test), len(child_inds_test))
    if args.max_samples is not None:
        n_eval = min(n_eval, args.max_samples)

    adult_inds_test = adult_inds_test[:n_eval]
    child_inds_test = child_inds_test[:n_eval]

    alae_model = load_model(
        "light_sbb/alae/ffhq.yaml",
        training_artifacts_dir=args.alae_artifacts_dir,
    )

    model, model_inv = load_transport_models(
        beta=args.beta,
        eps=args.eps,
        dim=args.dim,
        n_potentials=args.n_potentials,
        s_init=args.s_init,
        device=device,
    )

    adult_latents = torch.tensor(test_latents[adult_inds_test], dtype=torch.float32)
    generated_child_latents = map_adult_latents_to_child(model, model_inv, args.beta, adult_latents, device=device)
    generated_images = decode_latents(alae_model, generated_child_latents, batch_size=args.batch_size)

    real_child_images = test_images[child_inds_test].astype(np.uint8)

    bal_acc = compute_balanced_accuracy(
        real_child_images=real_child_images,
        generated_images=generated_images,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
    )

    with tempfile.TemporaryDirectory(prefix="alae_metrics_") as tmp_dir:
        tmp_dir = Path(tmp_dir)
        real_dir = tmp_dir / "real_child"
        gen_dir = tmp_dir / "generated_child"

        save_images_to_dir(real_child_images, real_dir)
        save_images_to_dir(generated_images, gen_dir)

        fid_value = fid_score.calculate_fid_given_paths(
            [str(real_dir), str(gen_dir)],
            batch_size=args.batch_size,
            device=device,
            dims=2048,
            num_workers=0,
        )

    print(f"Samples used: {n_eval}")
    print(f"Balanced accuracy (real child vs generated child): {bal_acc:.6f}")
    print(f"FID (generated child vs real child): {fid_value:.6f}")


if __name__ == "__main__":
    main()
