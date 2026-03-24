import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from light_sbb.alae.alae_ffhq_inference import decode, load_model
from light_sbb.lightsbm import LightSBM, MLP_network


def get_test_indices(age: np.ndarray, train_size: int = 60000, test_size: int = 10000):
    """Replicate the adult/child split from run_alae.py on the test partition."""
    test_age = age[train_size:]

    x_inds_test = np.arange(test_size)[
        (test_age >= 18).reshape(-1) * (test_age != -1).reshape(-1)
    ]
    y_inds_test = np.arange(test_size)[
        (test_age < 18).reshape(-1) * (test_age != -1).reshape(-1)
    ]
    return x_inds_test, y_inds_test


def load_transport(beta: float, eps: float, dim: int, n_potentials: int, s_init: float, device: torch.device):
    model = LightSBM(
        dim=dim,
        n_potentials=n_potentials,
        epsilon=eps,
        S_diagonal_init=s_init,
        is_diagonal=True,
    ).to(device)

    model_path = Path(f"model/b{beta}_e{eps}.pkl")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing transport checkpoint: {model_path}")
    with open(model_path, "rb") as f:
        model.load_state_dict(pickle.load(f), strict=True)
    model.eval()

    model_inv = None
    if beta < 100:
        inv_path = Path(f"model_inv/b{beta}_e{eps}.pkl")
        if not inv_path.exists():
            raise FileNotFoundError(f"Missing inverse checkpoint: {inv_path}")
        model_inv = MLP_network(input_dim=dim, t_model=32, d_model=128).to(device)
        with open(inv_path, "rb") as f:
            model_inv.load_state_dict(pickle.load(f), strict=True)
        model_inv.eval()

    return model, model_inv


@torch.no_grad()
def decode_latent_batch(alae_model, latents: torch.Tensor, batch_size: int = 64):
    imgs = []
    for i in range(0, len(latents), batch_size):
        decoded = decode(alae_model, latents[i:i + batch_size].cpu())
        decoded = ((decoded * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255)
        decoded = decoded.type(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        imgs.append(decoded)
    return np.concatenate(imgs, axis=0)


def save_png_sequence(images: np.ndarray, folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images, start=1):
        Image.fromarray(img).save(folder / f"{i:05d}.png")


@torch.no_grad()
def sample_adult_to_child_like_sample_alae(model, model_inv, beta: float, adult_test_latents: torch.Tensor,
                                           device: torch.device):
    """Use the same latent mapping rules as sample_alae, but for all adult test samples."""
    adult_test_latents = adult_test_latents.to(device)

    if beta >= 100:
        y_latent_to_map = adult_test_latents - 1.0 / beta * model.get_drift(
            torch.zeros(len(adult_test_latents), device=device), adult_test_latents
        )
    else:
        y_latent_to_map = model_inv(
            torch.zeros((len(adult_test_latents), 1), device=device),
            adult_test_latents,
        )

    return model(y_latent_to_map)


def export_test_image_folders(beta: float = 1.0,
                              eps: float = 0.1,
                              dim: int = 512,
                              n_potentials: int = 10,
                              s_init: float = 0.1,
                              batch_size: int = 64,
                              alae_artifacts_dir: str = "light_sbb/alae/training_artifacts/ffhq/",
                              real_out_dir: str = "real_child_test",
                              sbb_out_dir: str = "sbb_child_test"):
    """
    Build two folders from the test split:
    - real_child_test/: decoded images for every child in test set
    - sbb_child_test/: child translations for every adult in test set
    Each folder uses 5-digit numbering: 00001.png, 00002.png, ...
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latents = np.load("data/latents.npy")
    age = np.load("data/age.npy")

    test_latents = latents[60000:]
    x_inds_test, y_inds_test = get_test_indices(age)

    alae_model = load_model("light_sbb/alae/ffhq.yaml", training_artifacts_dir=alae_artifacts_dir)
    model, model_inv = load_transport(beta, eps, dim, n_potentials, s_init, device)

    # Real child test images from child latents decoded by ALAE.
    child_test_latents = torch.tensor(test_latents[y_inds_test], dtype=torch.float32)
    real_child_decoded = decode_latent_batch(alae_model, child_test_latents, batch_size=batch_size)
    save_png_sequence(real_child_decoded, Path(real_out_dir))

    # SBB-generated child images from every adult test latent (sample_alae mapping logic).
    adult_test_latents = torch.tensor(test_latents[x_inds_test], dtype=torch.float32)
    generated_child_latents = sample_adult_to_child_like_sample_alae(
        model=model,
        model_inv=model_inv,
        beta=beta,
        adult_test_latents=adult_test_latents,
        device=device,
    )
    sbb_child_decoded = decode_latent_batch(alae_model, generated_child_latents, batch_size=batch_size)
    save_png_sequence(sbb_child_decoded, Path(sbb_out_dir))

    print(f"Saved {len(real_child_decoded)} images to {real_out_dir}/")
    print(f"Saved {len(sbb_child_decoded)} images to {sbb_out_dir}/")


def parse_args():
    parser = argparse.ArgumentParser(description="Export real child and SBB child test folders as PNG sequences.")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--n-potentials", type=int, default=10)
    parser.add_argument("--s-init", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--alae-artifacts-dir", type=str, default="light_sbb/alae/training_artifacts/ffhq/")
    parser.add_argument("--real-out-dir", type=str, default="real_child_test")
    parser.add_argument("--sbb-out-dir", type=str, default="sbb_child_test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_test_image_folders(
        beta=args.beta,
        eps=args.eps,
        dim=args.dim,
        n_potentials=args.n_potentials,
        s_init=args.s_init,
        batch_size=args.batch_size,
        alae_artifacts_dir=args.alae_artifacts_dir,
        real_out_dir=args.real_out_dir,
        sbb_out_dir=args.sbb_out_dir,
    )
