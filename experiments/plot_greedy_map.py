from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILE = Path("results/data/fragility_map_epsgreedy.csv")
OUTPUT_DIR = Path("results/figures")

OUT_OPTIMAL = OUTPUT_DIR / "fragility_map_epsgreedy_optimal.pdf"
OUT_LOCKIN = OUTPUT_DIR / "fragility_map_epsgreedy_lockin.pdf"


def _plot_heatmap(
    data: pd.DataFrame,
    value_col: str,
    title: str,
    cbar_label: str,
    out_file: Path,
) -> None:
    pivot = data.pivot(index="gap", columns="delta", values=value_col)
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    x_vals = pivot.columns.to_numpy()
    y_vals = pivot.index.to_numpy()
    z = pivot.to_numpy()

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    im = ax.imshow(
        z,
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

    ax.set_xlabel("Exposure distortion ($\\delta$)")
    ax.set_ylabel("Reward gap ($\\Delta$)")
    ax.set_title(title)

    ax.set_xticks(np.arange(-0.5, len(x_vals), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_vals), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.35, alpha=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT_FILE)

    _plot_heatmap(
        data=data,
        value_col="p_optimal",
        title="Epsilon-greedy: Probability of optimal convergence",
        cbar_label="Convergence probability",
        out_file=OUT_OPTIMAL,
    )

    _plot_heatmap(
        data=data,
        value_col="p_lock_in",
        title="Epsilon-greedy: Probability of informational lock-in",
        cbar_label="Lock-in probability",
        out_file=OUT_LOCKIN,
    )

    print(f"Saved: {OUT_OPTIMAL}")
    print(f"Saved: {OUT_LOCKIN}")


if __name__ == "__main__":
    main()