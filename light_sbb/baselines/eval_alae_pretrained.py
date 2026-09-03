"""Score a LightSBB-M checkpoint trained elsewhere on the shared ALAE evaluation.

The models behind the published figures were trained on the internal cluster and
live in S3, not in this repo's ``artifacts/``. This script loads such a pair of
state dicts the way the original notebook does, then hands the transport to the
same scoring code every baseline goes through, so the row it produces is
comparable with the rest of the table.

The S3 credentials are deliberately absent. Fill in ``load_credentials`` below
with the block the notebook uses before running this.
"""

import argparse
import sys
from pathlib import Path

import torch

LIGHT_SBB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIGHT_SBB))
sys.path.insert(0, str(LIGHT_SBB / "alae"))

from baselines.alae_data import DIM  # noqa: E402
from baselines.eval_alae_baseline import (load_source_latents,  # noqa: E402
                                          score_transport)
from baselines.lightsbb import LARGE_BETA, SAFE_T, LightSBB  # noqa: E402
from lightsbm import LightSBM, MLP_network  # noqa: E402

MODEL_KEY = "gmlab_equities_time_series_generation_dev:sbbts/pickle/models/light_sb_alae"
INV_KEY = f"{MODEL_KEY}_inv"


def load_credentials():
    """Authenticate against the S3 buckets holding the pretrained checkpoints.

    Left unimplemented on purpose so no credentials enter version control. Paste
    the notebook's ``credentials`` dict and ``GMLAB_ENVIRONMENT.load_credentials``
    call here before running.
    """
    raise NotImplementedError("paste the notebook's credentials block here")


def load_pretrained(path, beta, eps, n_potentials, s_init, safe_t, device):
    """Rebuild the bridge and its inverse network from the S3 state dicts.

    Args:
        path: Suffix identifying the run, e.g. ``_b1_e1_test_final_K5.pkl``.
        beta: Beta the checkpoint was trained at, fixing the inference regime.
        eps: Entropic regularization used to build the model, overridden by the
            value stored in the checkpoint.
        n_potentials: Number of Gaussian potentials in the mixture.
        s_init: Initial covariance diagonal, overwritten by the loaded weights.
        safe_t: Time margin at which the Bass map is evaluated.
        device: Device the models are moved to.

    Returns:
        A ``LightSBB`` holding the loaded weights.
    """
    from gmlab_utils.gmlab_s3 import pickle_load

    model = LightSBM(dim=DIM, n_potentials=n_potentials, epsilon=eps,
                     S_diagonal_init=s_init, is_diagonal=True)
    model.load_state_dict(pickle_load(MODEL_KEY + path))
    model.to(device)

    # epsilon is a buffer, so the checkpoint overrides the flag. Report what was
    # actually loaded rather than what was asked for.
    eps = float(model.epsilon)

    baseline = LightSBB(input_dim=DIM, beta=beta, eps=eps, n_potentials=n_potentials,
                        s_init=s_init, safe_t=safe_t, device=device)
    baseline.model = model

    if baseline.model_inv is not None:
        model_inv = MLP_network(input_dim=DIM, t_model=32, d_model=128)
        model_inv.load_state_dict(pickle_load(INV_KEY + path))
        model_inv.to(device)
        baseline.model_inv = model_inv

    return baseline


def parse_args():
    """Parse the checkpoint identifiers and the shared evaluation settings."""
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True,
                   help="S3 key suffix, e.g. _b1_e1_test_final_K5.pkl")
    p.add_argument("--beta", type=float, required=True,
                   help=f"beta the checkpoint was trained at; at or above "
                        f"{LARGE_BETA:g} the inverse network is not used")
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--n-potentials", type=int, default=10)
    p.add_argument("--s-init", type=float, default=0.1)
    p.add_argument("--safe-t", type=float, default=SAFE_T)
    p.add_argument("--tag", help="results folder name; derived from --path otherwise")

    p.add_argument("--n-images", type=int, default=1000)
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--decode-batch", type=int, default=8,
                   help="latents decoded per ALAE call; lower it if memory is tight")
    p.add_argument("--threads", type=int, default=8,
                   help="CPU threads torch, InsightFace and OpenCV may use")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--no-fid", action="store_true",
                   help="skip the FID pass, which decodes the reference set once")
    p.add_argument("--n-reference", type=int, default=1000,
                   help="real child faces the FID is measured against")
    p.add_argument("--device", default="cuda:3")

    args = p.parse_args()
    # The pretrained checkpoint carries no seed of its own; the evaluation is
    # deterministic given the fixed held-out latents, so this only names the run.
    args.seed = 0
    args.model = LightSBB.name
    return args


def main():
    """Load a pretrained bridge from S3 and run the shared ALAE evaluation."""
    args = parse_args()

    torch.set_num_threads(args.threads)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(device)

    load_credentials()

    x_0 = load_source_latents(args)
    print(f"transporting {len(x_0)} held-out source latents")

    model = load_pretrained(args.path, args.beta, args.eps, args.n_potentials,
                            args.s_init, args.safe_t, device)
    x_1 = model.transport(x_0).cpu()

    tag = args.tag or f"lightsbb_pretrained{Path(args.path).stem}"
    print(score_transport(args, tag, x_0, x_1, model.checkpoint(), device))


if __name__ == "__main__":
    main()
