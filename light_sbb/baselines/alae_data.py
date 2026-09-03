"""Shared ALAE latent split and output paths for the baseline scripts."""

import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DIM = 512
TRAIN_SIZE = 60000
TEST_SIZE = 10000
MODEL_DIR = ROOT / "artifacts" / "alae_baselines"


def run_tag(args):
    """Name a run by the hyperparameters that distinguish it from a sibling.

    Only the bridges sweep beta and eps, so the other baselines keep the bare
    model name and the results they already produced stay addressable.

    Args:
        args: Parsed CLI arguments, carrying ``model`` and any swept values.

    Returns:
        Tag used for the checkpoint, the decoded images and the results folder.
    """
    tag = args.model
    if getattr(args, "beta", None) is not None:
        tag += f"_b{args.beta:g}"
    if getattr(args, "eps", None) is not None:
        tag += f"_e{args.eps:g}"
    return tag


def load_alae_splits(data_dir):
    """Load ALAE latents and split them into adult/child train and test sets.

    Mirrors the inline split in ``run_alae.py`` so baselines see the same data as
    LightSBB-M: adults are the source, children the target, and unlabelled faces
    (age == -1) are excluded from both.

    Args:
        data_dir: Directory holding ``latents.npy`` and ``age.npy``.

    Returns:
        Dict with the training latents, the test latents and the test source indices.
    """
    latents = np.load(os.path.join(data_dir, "latents.npy"))
    age = np.load(os.path.join(data_dir, "age.npy"))

    train_latents, test_latents = latents[:TRAIN_SIZE], latents[TRAIN_SIZE:]
    train_age, test_age = age[:TRAIN_SIZE], age[TRAIN_SIZE:]

    x_inds_train = np.arange(TRAIN_SIZE)[
        (train_age >= 18).reshape(-1) * (train_age != -1).reshape(-1)
    ]
    x_inds_test = np.arange(TEST_SIZE)[
        (test_age >= 18).reshape(-1) * (test_age != -1).reshape(-1)
    ]
    y_inds_train = np.arange(TRAIN_SIZE)[
        (train_age < 18).reshape(-1) * (train_age != -1).reshape(-1)
    ]
    y_inds_test = np.arange(TEST_SIZE)[
        (test_age < 18).reshape(-1) * (test_age != -1).reshape(-1)
    ]

    return {
        "x_train": train_latents[x_inds_train],
        "y_train": train_latents[y_inds_train],
        "test_latents": test_latents,
        "x_inds_test": x_inds_test,
        "y_inds_test": y_inds_test,
    }
