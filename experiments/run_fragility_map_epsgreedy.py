import csv
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from src.robustness_checks import run_episode_epsilon_greedy


MU0 = 0.60
HORIZON = 5_000
N_RUNS = 100
EPSILON = 0.10

DELTA_VALUES = [0.00, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
GAP_VALUES = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]

OUTPUT_DIR = Path("results/data")
OUTPUT_FILE = OUTPUT_DIR / "fragility_map_epsgreedy.csv"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    total_cells = len(DELTA_VALUES) * len(GAP_VALUES)
    progress = tqdm(total=total_cells, desc="Running eps-greedy fragility grid")

    for gap in GAP_VALUES:
        mu1 = MU0 - gap

        for delta in DELTA_VALUES:
            outcomes = []

            for seed in range(N_RUNS):
                result = run_episode_epsilon_greedy(
                    mu_0=MU0,
                    mu_1=mu1,
                    delta=delta,
                    horizon=HORIZON,
                    seed=seed,
                    epsilon=EPSILON,
                    inferior_arm=1,
                )
                outcomes.append(result.outcome)

            counts = Counter(outcomes)
            p_optimal = counts.get("optimal_convergence", 0) / N_RUNS
            p_lock_in = counts.get("lock_in", 0) / N_RUNS
            p_ambiguous = counts.get("ambiguous", 0) / N_RUNS

            rows.append(
                {
                    "algorithm": "epsilon_greedy",
                    "epsilon": EPSILON,
                    "delta": float(delta),
                    "gap": float(gap),
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
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algorithm",
                "epsilon",
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