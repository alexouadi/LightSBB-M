"""Build the qualitative ALAE comparison: one column per method, one row per face.

Reads the PNG folders ``eval_alae_baseline.py`` leaves behind, so it needs no
model, no decoder and no GPU. Every method is shown on the same held-out inputs,
which is what makes the columns comparable.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "artifacts" / "alae_baseline_images"
FIGURE_DIR = ROOT / "figures" / "alae"

# Column labels as they should read in the paper, not the folder names.
LABELS = {
    "input": "Input",
    "otcfm": "OT-CFM",
    "not": "NOT",
    "sf2m": "[SF]$^2$M",
    "lightsb": "LightSB-M",
    "lightsbb": "LightSBB-M",
}

# The baselines carry a seed; the pretrained bridges are keyed by beta instead.
SEEDED = ("otcfm", "not", "sf2m")


def bridge_folder(model, beta, eps, test, n_images):
    """Name the image folder of one pretrained bridge.

    Args:
        model: Either ``lightsbb`` or ``lightsb``.
        beta: Beta that bridge was trained at.
        eps: Epsilon that bridge was trained at.
        test: Suffix identifying the checkpoint, as in the S3 key.
        n_images: Size of the held-out set the folder was decoded for.

    Returns:
        The folder name written by the pretrained evaluation.
    """
    return f"{model}_pretrained_b{beta:g}_e{eps:g}_test_{test}_s0_n{n_images}"


def folder_names(sbb, sb, eps, seeds, n_images):
    """Map each column to the image folder holding its decoded outputs.

    The two bridges were trained separately, so each carries its own beta and
    checkpoint suffix rather than sharing one setting.

    Args:
        sbb: (beta, test) of the pretrained LightSBB-M checkpoint.
        sb: (beta, test) of the pretrained LightSB-M checkpoint.
        eps: Epsilon both bridges were trained at.
        seeds: Dict giving the seed of each trained-here baseline.
        n_images: Size of the held-out set the folders were decoded for.

    Returns:
        Dict mapping column key to folder name, in display order.
    """
    return {
        "input": f"input_n{n_images}",
        **{name: f"{name}_s{seeds[name]}_n{n_images}" for name in SEEDED},
        "lightsb": bridge_folder("lightsb", sb[0], eps, sb[1], n_images),
        "lightsbb": bridge_folder("lightsbb", sbb[0], eps, sbb[1], n_images),
    }


def load_column(folder, indices):
    """Load one method's images for the selected faces.

    Args:
        folder: Directory of decoded PNGs, named by index.
        indices: Indices into the held-out set.

    Returns:
        List of images, one per index.
    """
    images = []
    for i in indices:
        path = folder / f"{i:05d}.png"
        if not path.exists():
            raise FileNotFoundError(path)
        images.append(Image.open(path))
    return images


def build_figure(columns, indices, out_path, label_size=9):
    """Draw the grid and write it to ``out_path``.

    Args:
        columns: Dict mapping column label to its list of images.
        indices: Indices being shown, used only for the row labels.
        out_path: Destination file, its parents created as needed.
        label_size: Font size of the column headers.

    Returns:
        The path written.
    """
    n_rows, n_cols = len(indices), len(columns)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols, n_rows + 0.35), dpi=200,
                             squeeze=False)

    for col, (label, images) in enumerate(columns.items()):
        for row, image in enumerate(images):
            ax = axes[row][col]
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)

        axes[0][col].set_title(label, fontsize=label_size, pad=4)

    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_path


def parse_args():
    """Parse the faces to show and the checkpoint each bridge column comes from."""
    p = argparse.ArgumentParser()
    p.add_argument("--indices", type=int, nargs="+", default=[3, 72, 137, 824],
                   help="indices into the held-out set, one row each")
    p.add_argument("--beta-sbb", type=float, default=1.0,
                   help="beta of the pretrained LightSBB-M column")
    p.add_argument("--beta-sb", type=float, default=1.0,
                   help="beta of the pretrained LightSB-M column")
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--test-sbb", default="final_K5",
                   help="LightSBB-M checkpoint suffix, the part after test_")
    p.add_argument("--test-sb", default="final_K5",
                   help="LightSB-M checkpoint suffix, the part after test_")
    p.add_argument("--seed-otcfm", type=int, default=1,
                   help="seed of the OT-CFM column")
    p.add_argument("--seed-not", type=int, default=0,
                   help="seed of the NOT column")
    p.add_argument("--seed-sf2m", type=int, default=0,
                   help="seed of the [SF]^2M column")
    p.add_argument("--n-images", type=int, default=1000)
    p.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    p.add_argument("--out", type=Path,
                   help="output file; derived from beta and eps otherwise")
    return p.parse_args()


def main():
    """Assemble the comparison figure from the decoded image folders."""
    args = parse_args()

    names = folder_names((args.beta_sbb, args.test_sbb),
                         (args.beta_sb, args.test_sb),
                         args.eps,
                         {"otcfm": args.seed_otcfm, "not": args.seed_not,
                          "sf2m": args.seed_sf2m},
                         args.n_images)
    columns = {}
    for key, name in names.items():
        folder = args.image_dir / name
        if not folder.is_dir():
            raise FileNotFoundError(f"{folder} is missing; evaluate {key} first")
        columns[LABELS[key]] = load_column(folder, args.indices)

    out = args.out or (FIGURE_DIR /
                       f"alae_comparison_sbb_b{args.beta_sbb:g}"
                       f"_sb_b{args.beta_sb:g}_e{args.eps:g}.png")
    print(build_figure(columns, args.indices, out))


if __name__ == "__main__":
    main()
