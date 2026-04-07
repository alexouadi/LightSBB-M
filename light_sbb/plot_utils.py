"""
plot_utils.py — publication-quality distribution comparison plots for LightSBB experiments.

Primary entry point:
    plot_distributions(arrays, labels, ...)

Typical usage — comparing any number of distributions on the same axes:
    samples_a = np.load("samples_config_a.npy")
    samples_b = np.load("samples_config_b.npy")
    real      = np.load("real_data.npy")
    plot_distributions(
        [samples_a, samples_b, real],
        ["LightSBB  β=10", "LightSBB  β=50", "Real data"],
        xlabel=r"$x$",
        savepath="comparison.pdf",
    )
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import gaussian_kde
from typing import List, Optional, Tuple


# Colorblind-friendly palette (ColorBrewer RdBu + extras)
_COLORS = ["#2166AC", "#D6604D", "#4DAC26", "#8073AC", "#E08214"]

_PAPER_RC = {
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.borderpad": 0.4,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean(arr: np.ndarray) -> np.ndarray:
    """Flatten to 1-D and drop non-finite values."""
    arr = np.asarray(arr, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def _auto_xlim(
    arrays: List[np.ndarray],
    quantile_clip: float,
) -> Tuple[float, float]:
    """Derive x limits from joint empirical quantiles across all arrays."""
    lo = min(np.quantile(a, quantile_clip / 2) for a in arrays)
    hi = max(np.quantile(a, 1 - quantile_clip / 2) for a in arrays)
    pad = (hi - lo) * 0.06
    return lo - pad, hi + pad


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_distributions(
    arrays: List[np.ndarray],
    labels: List[str],
    *,
    # --- labels / legend ---
    show_n: bool = True,
    # --- axes ---
    xlabel: str = r"$x$",
    ylabel: str = "Density",
    xlim: Optional[Tuple[float, float]] = None,
    quantile_clip: float = 0.02,
    # --- KDE ---
    bw_method: Optional[float] = None,
    n_eval: int = 1024,
    # --- histogram ---
    show_hist: bool = False,
    n_bins: int = 60,
    # --- rug plot ---
    show_rug: bool = False,
    # --- appearance ---
    colors: Optional[List[str]] = None,
    linewidth: float = 1.8,
    fill_alpha: float = 0.15,
    grid: bool = True,
    figsize: Tuple[float, float] = (4.5, 3.0),
    fontsize: float = 9.0,
    # --- reference Gaussian overlay ---
    show_gaussian_ref: bool = False,
    gaussian_ref_label: str = r"$\mathcal{N}(0,1)$",
    # --- output ---
    savepath: Optional[str] = None,
    dpi: int = 300,
    show: bool = False,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot KDE-estimated densities of any number of 1-D sample arrays on the same axes.

    Parameters
    ----------
    arrays : list of array-like
        Sample arrays (any shape; flattened internally).  Non-finite values are
        silently dropped.
    labels : list of str
        Legend labels, one per array.  Must have the same length as ``arrays``.
    show_n : bool
        Append ``(n=...)`` to each legend label (default True).
    xlabel, ylabel : str
        Axis labels.  LaTeX math strings are supported.
    xlim : (float, float) or None
        x-axis limits.  When None, derived automatically from joint quantiles
        (controlled by ``quantile_clip``).
    quantile_clip : float
        Fraction of probability mass clipped from each tail when computing
        automatic x limits.  Default 0.02 (clips outermost 2 %).
    bw_method : float or None
        Bandwidth passed to ``scipy.stats.gaussian_kde``.  None uses Scott's
        rule (scipy default).
    n_eval : int
        Number of points at which the KDE is evaluated.
    show_hist : bool
        Overlay a normalised histogram beneath the KDE lines.
    n_bins : int
        Number of histogram bins (used only when ``show_hist=True``).
    show_rug : bool
        Draw a rug plot along the x-axis.
    colors : list of str or None
        Hex / named colours, one per array.  Defaults to a colorblind-friendly
        palette (cycles if more arrays than colours are provided).
    linewidth : float
        KDE line width.
    fill_alpha : float
        Opacity of the KDE fill.
    figsize : (float, float)
        Figure dimensions in inches.
    fontsize : float
        Base font size; tick labels are ``fontsize - 1``.
    show_gaussian_ref : bool
        Overlay a standard N(0,1) reference density (useful for heavy-tail
        comparisons).
    gaussian_ref_label : str
        Legend label for the Gaussian reference.
    savepath : str or None
        If given, the figure is saved to this path.
    dpi : int
        Resolution used when saving.
    show : bool
        Call ``plt.show()`` after plotting.
    ax : matplotlib.axes.Axes or None
        Existing axes to draw into.  When None a new figure is created.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if len(arrays) != len(labels):
        raise ValueError(
            f"arrays and labels must have the same length "
            f"(got {len(arrays)} arrays and {len(labels)} labels)."
        )

    rc = {
        **_PAPER_RC,
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize,
        "xtick.labelsize": fontsize - 1,
        "ytick.labelsize": fontsize - 1,
        "legend.fontsize": fontsize - 1,
    }

    with plt.rc_context(rc):
        cleaned = [_clean(a) for a in arrays]
        palette = colors if colors is not None else _COLORS

        if xlim is None:
            xlim = _auto_xlim(cleaned, quantile_clip)

        xs = np.linspace(xlim[0], xlim[1], n_eval)
        kdes = [gaussian_kde(a, bw_method=bw_method) for a in cleaned]
        ys = [kde(xs) for kde in kdes]

        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # --- histogram (optional, drawn first so KDE sits on top) ---
        if show_hist:
            bins = np.linspace(xlim[0], xlim[1], n_bins + 1)
            for a, c in zip(cleaned, palette):
                ax.hist(a, bins=bins, density=True,
                        color=c, alpha=0.22, linewidth=0, zorder=1)

        # --- KDE fill + line ---
        for a, y, lbl, c in zip(cleaned, ys, labels, palette):
            legend_label = f"{lbl}  (n={len(a):,})" if show_n else lbl
            ax.fill_between(xs, y, alpha=fill_alpha, color=c, zorder=2)
            ax.plot(xs, y, color=c, linewidth=linewidth, label=legend_label, zorder=3)

        # --- optional Gaussian reference ---
        if show_gaussian_ref:
            from scipy.stats import norm as _norm
            y_ref = _norm.pdf(xs)
            ax.plot(xs, y_ref, color="0.55", linewidth=linewidth * 0.85,
                    linestyle="--", label=gaussian_ref_label, zorder=3)

        # --- rug plot (optional) ---
        if show_rug:
            y_max = max(y.max() for y in ys)
            rug_height = 0.018 * y_max
            for i, (a, c) in enumerate(zip(cleaned, palette)):
                rug_offset = -0.025 * y_max - i * rug_height * 0.5
                a_clipped = np.clip(a, *xlim)
                ax.plot(a_clipped,
                        np.full_like(a_clipped, rug_offset),
                        marker="|", linestyle="none",
                        color=c, alpha=0.25, markersize=rug_height * 100,
                        markeredgewidth=0.6, zorder=4)

        # --- axes decoration ---
        ax.set_xlim(xlim)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

        if grid:
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)

        ax.legend(loc="best")

        if own_fig:
            fig.tight_layout(pad=0.4)

        if savepath is not None:
            fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

        if show:
            plt.show()
            plt.close(fig)  # prevent Jupyter inline backend from showing it a second time

    return fig
