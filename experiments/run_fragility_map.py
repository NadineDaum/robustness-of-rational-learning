from collections import Counter
from pathlib import Path
import csv

from tqdm import tqdm

from src.simulation import run_episode


# -----------------------------
# Experiment parameters
# -----------------------------
# We use Option A parameterization: keep mu_0 fixed and vary the gap.
MU0 = 0.60
HORIZON = 5_000
N_RUNS = 100

DELTA_VALUES = [0.00, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
GAP_VALUES = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]

OUTPUT_DIR = Path("results/data")
OUTPUT_FILE = OUTPUT_DIR / "fragility_map.csv"


def main() -> None:
    """
    Run the main fragility-map experiment.

    For each (delta, gap) pair, simulate repeated learning trajectories and
    estimate the probabilities of:
    - optimal convergence
    - informational lock-in
    - ambiguous learning
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # One row per (delta, gap) cell for easy plotting later.
    rows = []

    total_cells = len(DELTA_VALUES) * len(GAP_VALUES)
    progress = tqdm(total=total_cells, desc="Running parameter grid")

    for gap in GAP_VALUES:
        # Smaller gap means a harder identification problem for the learner.
        mu1 = MU0 - gap

        for delta in DELTA_VALUES:
            outcomes = []

            for seed in range(N_RUNS):
                # Controlled seeds make reruns deterministic and comparable.
                result = run_episode(
                    mu_0=MU0,
                    mu_1=mu1,
                    delta=delta,
                    horizon=HORIZON,
                    seed=seed,
                )
                outcomes.append(result.outcome)

            counts = Counter(outcomes)

            # Convert raw counts into empirical probabilities for this cell.
            p_optimal = counts.get("optimal_convergence", 0) / N_RUNS
            p_lock_in = counts.get("lock_in", 0) / N_RUNS
            p_ambiguous = counts.get("ambiguous", 0) / N_RUNS

            rows.append(
                {
                    "delta": delta,
                    "gap": gap,
                    "mu_0": MU0,
                    "mu_1": mu1,
                    "horizon": HORIZON,
                    "n_runs": N_RUNS,
                    "p_optimal": p_optimal,
                    "p_lock_in": p_lock_in,
                    "p_ambiguous": p_ambiguous,
                }
            )

            progress.update(1)

    progress.close()

    with OUTPUT_FILE.open("w", newline="") as f:
        # Flat CSV keeps downstream plotting and thesis tables simple.
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "delta",
                "gap",
                "mu_0",
                "mu_1",
                "horizon",
                "n_runs",
                "p_optimal",
                "p_lock_in",
                "p_ambiguous",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()