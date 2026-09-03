"""Train an unpaired transport baseline on ALAE latents (adult -> child)."""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines import BASELINES  # noqa: E402
from baselines.alae_data import DIM, MODEL_DIR, load_alae_splits, run_tag  # noqa: E402
from utils import TensorSampler  # noqa: E402


def parse_args():
    """Parse the shared arguments plus the selected baseline's own flags."""
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(BASELINES), default="otcfm")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--threads", type=int, default=8,
                   help="CPU threads torch may use")
    p.add_argument("--device", default="cuda:3")

    known, _ = p.parse_known_args()
    BASELINES[known.model].add_arguments(p)
    return p.parse_args()


def main():
    """Train the requested baseline and pickle its checkpoint."""
    args = parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(device)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    splits = load_alae_splits(args.data_dir)
    print(f"train source {len(splits['x_train'])}, train target {len(splits['y_train'])}")

    X_sampler = TensorSampler(torch.tensor(splits["x_train"]), device=device)
    Y_sampler = TensorSampler(torch.tensor(splits["y_train"]), device=device)

    model = BASELINES[args.model].from_args(input_dim=DIM, args=args, device=device)
    model.train(X_sampler, Y_sampler, args)

    save_path = MODEL_DIR / f"{run_tag(args)}_s{args.seed}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(model.checkpoint(), f)

    print(save_path)


if __name__ == "__main__":
    main()
