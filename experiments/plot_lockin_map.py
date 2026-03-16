from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_FILE = Path("results/data/fragility_map.csv")
OUTPUT_DIR = Path("results/figures")
OUTPUT_FILE = OUTPUT_DIR / "fragility_map_lockin.pdf"


def main() -> None:
    """
    Create the companion thesis figure:
    probability of informational lock-in over exposure distortion and reward gap.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load grid-level probabilities produced by run_fragility_map.py.
    data = pd.read_csv(INPUT_FILE)

    # Re-shape long table into matrix form for heatmap plotting.
    pivot = data.pivot(index="gap", columns="delta", values="p_lock_in")
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    x_vals = pivot.columns.to_numpy()
    y_vals = pivot.index.to_numpy()
    z = pivot.to_numpy()

    # Match layout with the optimal map for side-by-side readability.
    fig, ax = plt.subplots(figsize=(7.6, 5.4))

    # Main heatmap
    im = ax.imshow(
        z,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    # Axis ticks and labels
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([f"{x:.2f}" for x in x_vals], rotation=45, ha="right")
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels([f"{y:.2f}" for y in y_vals])

    ax.set_xlabel("Exposure distortion ($\\delta$)")
    ax.set_ylabel("Reward gap ($\\Delta$)")

    # Thin grid lines to show discrete parameter cells
    ax.set_xticks(np.arange(-0.5, len(x_vals), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_vals), 1), minor=True)
    # Subtle cell borders help show the discrete experiment grid.
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.35, alpha=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Lock-in probability")

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()