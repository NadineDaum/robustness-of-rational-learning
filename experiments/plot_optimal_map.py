from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_FILE = Path("results/data/fragility_map.csv")
OUTPUT_DIR = Path("results/figures")
OUTPUT_FILE = OUTPUT_DIR / "fragility_map_optimal.pdf"


def main() -> None:
    """
    Create the main thesis figure:
    probability of optimal convergence across exposure distortion
    and reward gap.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load grid-level probabilities produced by run_fragility_map.py.
    data = pd.read_csv(INPUT_FILE)

    # Re-shape long table into matrix form for imshow/contours.
    pivot = data.pivot(index="gap", columns="delta", values="p_optimal")
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    x_vals = pivot.columns.to_numpy()
    y_vals = pivot.index.to_numpy()
    z = pivot.to_numpy()

    # Slightly wider-than-tall figure works well for thesis page layout.
    fig, ax = plt.subplots(figsize=(7.6, 5.4))

    # Heatmap
    im = ax.imshow(
        z,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    # Axis ticks
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([f"{x:.2f}" for x in x_vals], rotation=45, ha="right")

    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels([f"{y:.2f}" for y in y_vals])

    # Axis labels
    ax.set_xlabel("Exposure distortion ($\\delta$)")
    ax.set_ylabel("Reward gap ($\\Delta$)")

    # Thin grid showing discrete parameter cells
    ax.set_xticks(np.arange(-0.5, len(x_vals), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_vals), 1), minor=True)

    ax.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=0.35,
        alpha=0.3,
    )

    ax.tick_params(which="minor", bottom=False, left=False)

    # Contour lines for convergence thresholds
    xx, yy = np.meshgrid(
        np.arange(len(x_vals)),
        np.arange(len(y_vals))
    )

    contours = ax.contour(
        xx,
        yy,
        z,
        levels=[0.5, 0.9],
        colors=["black", "black"],
        linewidths=[1.0, 1.2],
        linestyles=["--", "-"],
    )

    ax.clabel(
        contours,
        fmt={0.5: "0.5", 0.9: "0.9"},
        inline=True,
        fontsize=9,
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Convergence probability")

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()