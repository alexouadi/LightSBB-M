"""Build the ground-truth figure and the paper table from the saved run results.

Reads every metrics JSON and weight file under results/ground_truth/, writes the
control figure to figures/ground_truth/ and the booktabs table to
results/ground_truth/table.tex.

Run from inside light_sbb/ground_truth/:  python make_figures.py
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gaussian_ground_truth as gt  # noqa: E402
from control_extraction import SAFE_T, bridge_y, drift_x, sigma_x  # noqa: E402
from lightsbm import LightSBM  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "ground_truth"
FIGURES_DIR = ROOT / "figures" / "ground_truth"

# Publication palette: one colour per beta, plus a neutral for the baseline.
PALETTE = {2.0: "#0B5FA5", 10.0: "#C2410C", 100.0: "#15803D"}
BASELINE_COLOR = "#6B7280"
EXACT_STYLE = dict(color="#111827", linestyle="--", linewidth=1.4, zorder=3)

METRIC_COLUMNS = [("terminal_sw2", "terminal $SW_2$"), ("plan_sw2", "plan $SW_2$"),
                  ("cross_cov_err", "cross-cov."), ("objective_gap", "obj. gap"),
                  ("drift_only_gap", "drift gap"), ("E_a", "$E_a$"), ("E_b", "$E_b$")]


def load_rows(results_dir=RESULTS_DIR):
    """Collect every metric row written under the per-run folders.

    Args:
        results_dir: root holding <family>/seed_<s>/K_<k>/metrics.json.

    Returns:
        List of dicts, one per (method, beta, K, seed), each carrying `run_dir`.
    """
    rows = []
    for path in sorted(results_dir.glob("*/seed_*/K_*/metrics.json")):
        with open(path) as f:
            loaded = json.load(f)
        for r in loaded if isinstance(loaded, list) else [loaded]:
            r["run_dir"] = str(path.parent)
            rows.append(r)
    return rows


def weights_for(rows, beta, method="lightsbb", k_ref=None):
    """Find the checkpoint of one run.

    Args:
        rows: metric rows from `load_rows`.
        beta: beta to match.
        method: "lightsbb" or "lightsb".
        k_ref: preferred K; the largest available is used when absent.

    Returns:
        (Path, row) of the chosen run, or (None, None) if it has no weights.
    """
    match = [r for r in rows if r["method"] == method and r["beta"] == beta]
    if k_ref is not None and any(r["K"] == k_ref for r in match):
        match = [r for r in match if r["K"] == k_ref]
    if not match:
        return None, None
    row = max(match, key=lambda r: r["K"])
    path = Path(row["run_dir"]) / "weights.pt"
    return (path, row) if path.exists() else (None, None)


def load_model(path, eps=None, n_potentials=None, dim=2, device="cpu"):
    """Rebuild a trained LightSBM from a saved checkpoint.

    Args:
        path: .pt file written by the run script.
        eps: diffusion scale; taken from the checkpoint when absent.
        n_potentials: mixture size; taken from the checkpoint when absent.
        dim: data dimension.
        device: torch device.

    Returns:
        The model in eval mode.
    """
    state = torch.load(path, map_location=device)
    model = LightSBM(dim=dim,
                     n_potentials=n_potentials or state.get("n_potentials", 50),
                     epsilon=eps if eps is not None else state.get("eps", 1.0),
                     is_diagonal=True).to(device)
    model.load_state_dict(state["model"])
    return model.eval()


def control_curves(model, beta, times, n=4096, sigma2=(10.0, 0.1), eps=1.0, seed=0):
    """Average the learned drift ratio and volatility diagonal along the bridge.

    The drift is summarised as the fitted coefficient a_hat(t) obtained by projecting
    a_hat onto the state, which is what the exact solution parameterises.

    Args:
        model: trained LightSBM.
        beta: SBB volatility penalty.
        times: (m,) times in [0, 1).
        n: bridge samples per time.
        sigma2: target marginal variances.
        eps: diffusion scale.
        seed: RNG seed.

    Returns:
        (a_hat, b_hat) each of shape (m, d).
    """
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    r = gt.solution(beta, sigma2)["r"]

    y_0 = torch.randn(n, len(r), device=device)
    y_T = model(y_0).detach()

    a_out, b_out = [], []
    for t in times:
        y_t = bridge_y(y_0, y_T, t, torch.as_tensor(eps, device=device))
        t_vec = torch.full((n,), float(t), device=device)
        a = drift_x(model, t_vec, y_t, beta).detach()
        s = sigma_x(model, t_vec, y_t, beta).detach()
        # a*(t, x) = a(t) x per coordinate, so the least-squares slope recovers a(t).
        x = (y_t + model.get_drift(t_vec, y_t) / beta).detach()
        a_out.append(((a * x).mean(0) / (x**2).mean(0)).cpu().numpy())
        b_out.append(s.diagonal(dim1=-2, dim2=-1).mean(0).cpu().numpy())
    return np.array(a_out), np.array(b_out)


def panel_controls(ax, rows, times, kind, coord, sigma2, device, k_ref=None, eps=1.0):
    """Draw learned vs exact controls for one coordinate on one axis.

    Args:
        ax: matplotlib axis.
        rows: metric rows, used to find which betas have weights.
        times: (m,) times.
        kind: "drift" or "vol".
        coord: coordinate index to plot.
        sigma2: target marginal variances.
        device: torch device.
        k_ref: preferred K when several runs exist.
        eps: diffusion scale.
    """
    betas = sorted({r["beta"] for r in rows if r["method"] == "lightsbb"})
    for beta in betas:
        path, row = weights_for(rows, beta, "lightsbb", k_ref)
        if path is None:
            continue
        model = load_model(path, device=device)
        a_hat, b_hat = control_curves(model, beta, times, sigma2=sigma2, eps=eps)
        learned = (a_hat if kind == "drift" else b_hat)[:, coord]
        ax.plot(times, learned, color=PALETTE.get(beta, "#444"), linewidth=1.8,
                label=rf"$\beta = {beta:g}$ (learned)")

        r = gt.solution(beta, sigma2)["r"][coord]
        exact = ([gt.drift_coeff(t, r) for t in times] if kind == "drift"
                 else [gt.vol_coeff(t, r, beta) for t in times])
        ax.plot(times, exact, **EXACT_STYLE)

    ax.set_xlabel("$t$")
    ax.set_ylabel(rf"$a_{coord + 1}(t)$" if kind == "drift" else rf"$b_{coord + 1}(t)$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, frameon=False)


def aggregate(rows, method, beta, key, k=None):
    """Average one metric over seeds.

    Args:
        rows: metric rows.
        method: "lightsbb" or "lightsb".
        beta: beta to match.
        key: metric name.
        k: K to match, or None for any.

    Returns:
        (mean, std, count), or (nan, nan, 0) when nothing matches.
    """
    vals = [r[key] for r in rows if r["method"] == method and r["beta"] == beta
            and (k is None or r["K"] == k) and np.isfinite(r.get(key, np.nan))]
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def panel_sweep(ax, rows):
    """Draw plan SW2 against the number of outer iterations K, averaged over seeds."""
    betas = sorted({r["beta"] for r in rows if r["method"] == "lightsbb"})
    for beta in betas:
        ks = sorted({r["K"] for r in rows if r["method"] == "lightsbb" and r["beta"] == beta})
        if len(ks) < 2:
            continue
        stats = [aggregate(rows, "lightsbb", beta, "plan_sw2", k) for k in ks]
        mean = np.array([s[0] for s in stats])
        std = np.array([s[1] for s in stats])
        color = PALETTE.get(beta, "#444")
        ax.plot(ks, mean, marker="o", markersize=4, color=color, linewidth=1.8,
                label=rf"$\beta = {beta:g}$")
        if np.any(std > 0):
            ax.fill_between(ks, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)

    for beta in betas:
        mean, _, n = aggregate(rows, "lightsb", beta, "plan_sw2")
        if n:
            ax.axhline(mean, color=BASELINE_COLOR, linestyle=":", linewidth=1.4)
            ax.text(ax.get_xlim()[1], mean, " LightSB-M", fontsize=7,
                    color=BASELINE_COLOR, va="center")
            break

    ax.set_xlabel("outer iterations $K$")
    ax.set_ylabel(r"$SW_2(\hat\pi^K, \pi^*)$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, frameon=False)


def build_figure(rows, sigma2, device, n_times=41):
    """Assemble the control figure, one row per coordinate, and save it.

    Args:
        rows: metric rows.
        sigma2: target marginal variances.
        device: torch device.
        n_times: resolution of the control curves.

    Returns:
        Path to the saved figure.
    """
    times = np.linspace(0.0, 1.0 - SAFE_T, n_times)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0))

    for coord in range(len(sigma2)):
        panel_controls(axes[coord, 0], rows, times, "drift", coord, sigma2, device)
        panel_controls(axes[coord, 1], rows, times, "vol", coord, sigma2, device)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "ground_truth.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def build_table(rows, k_ref=5):
    """Write the booktabs results table, averaging each cell over seeds.

    Args:
        rows: metric rows.
        k_ref: the K whose LightSBB rows go into the table.

    Returns:
        Path to the .tex file.
    """
    betas = sorted({r["beta"] for r in rows})
    multi_seed = len({r["seed"] for r in rows}) > 1

    lines = [r"\begin{tabular}{ll" + "r" * len(METRIC_COLUMNS) + "}", r"\toprule",
             r"$\beta$ & method & " + " & ".join(h for _, h in METRIC_COLUMNS) + r" \\",
             r"\midrule"]

    for beta in betas:
        for method, label, k in (("lightsbb", "LightSBB-M", k_ref), ("lightsb", "LightSB-M", None)):
            cells = []
            for key, _ in METRIC_COLUMNS:
                mean, std, n = aggregate(rows, method, beta, key, k)
                if not n:
                    break
                cells.append(f"{mean:.3f}" + (rf"\,\pm\,{std:.3f}" if multi_seed and n > 1 else ""))
            if not cells:
                continue
            head = f"{beta:g}" if method == "lightsbb" else ""
            lines.append(f"{head} & {label} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "table.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sigma2", type=float, nargs=2, default=[10.0, 0.1])
    p.add_argument("--k-ref", type=int, default=5, help="K used for the table rows")
    p.add_argument("--n-times", type=int, default=41)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    rows = load_rows()
    if not rows:
        raise SystemExit(f"No metrics_*.json under {RESULTS_DIR}")
    print(f"Loaded {len(rows)} rows from {RESULTS_DIR}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fig_path = build_figure(rows, tuple(args.sigma2), device, args.n_times)
    tab_path = build_table(rows, args.k_ref)
    print(f"Figure -> {fig_path}")
    print(f"Table  -> {tab_path}")


if __name__ == "__main__":
    main()
