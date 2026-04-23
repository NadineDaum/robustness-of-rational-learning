from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.image import AxesImage


UCB1_FILE = Path("results/data/fragility_map_ucb1.csv")
EPS_FILE = Path("results/data/fragility_map_epsgreedy.csv")
OUTPUT_DIR = Path("results/figures")
OUTPUT_FILE = OUTPUT_DIR / "fragility_map_robustness_panel.pdf"


def _prepare_matrix(data: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = data.pivot(index="gap", columns="delta", values=value_col)
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    x_vals = pivot.columns.to_numpy()
    y_vals = pivot.index.to_numpy()
    z_vals = pivot.to_numpy()
    return x_vals, y_vals, z_vals


def _draw_heatmap(
    ax: plt.Axes,
    data: pd.DataFrame,
    value_col: str,
    show_x_label: bool,
    show_y_label: bool,
) -> AxesImage:
    x_vals, y_vals, z_vals = _prepare_matrix(data, value_col)

    im = ax.imshow(
        z_vals,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([f"{x:.2f}" for x in x_vals], rotation=45, ha="right")
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels([f"{y:.2f}" for y in y_vals])

    if show_x_label:
        ax.set_xlabel("Exposure distortion ($\\delta$)")
    else:
        ax.set_xlabel("")

    if show_y_label:
        ax.set_ylabel("Reward gap ($\\Delta$)")
    else:
        ax.set_ylabel("")

    ax.set_xticks(np.arange(-0.5, len(x_vals), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_vals), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.35, alpha=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ucb1 = pd.read_csv(UCB1_FILE)
    eps = pd.read_csv(EPS_FILE)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )

    im_left = _draw_heatmap(
        ax=axes[0, 0],
        data=ucb1,
        value_col="p_optimal",
        show_x_label=False,
        show_y_label=True,
    )

    _draw_heatmap(
        ax=axes[0, 1],
        data=ucb1,
        value_col="p_lock_in",
        show_x_label=False,
        show_y_label=False,
    )

    _draw_heatmap(
        ax=axes[1, 0],
        data=eps,
        value_col="p_optimal",
        show_x_label=True,
        show_y_label=True,
    )

    im_right = _draw_heatmap(
        ax=axes[1, 1],
        data=eps,
        value_col="p_lock_in",
        show_x_label=True,
        show_y_label=False,
    )

    cbar_left = fig.colorbar(im_left, ax=axes[:, 0], shrink=0.95)
    cbar_left.set_label("Convergence probability")

    cbar_right = fig.colorbar(im_right, ax=axes[:, 1], shrink=0.95)
    cbar_right.set_label("Lock-in probability")

    fig.savefig(OUTPUT_FILE, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
