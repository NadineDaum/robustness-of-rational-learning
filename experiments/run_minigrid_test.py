from collections import Counter
import numpy as np

from src.simulation import run_episode


# Quick sanity grid before launching heavier experiment runs.
MU0 = 0.6
HORIZON = 5_000
N_RUNS = 100
DELTA_VALUES = np.linspace(0.0, 0.20, 11)
GAP_VALUES = np.linspace(0.02, 0.20, 10)


def main() -> None:
    """
    Run a small parameter grid to verify that the model behaves
    in line with the theoretical mechanism before scaling up.
    """
    print("delta | gap | optimal | ambiguous | lock_in")
    print("-" * 48)

    for gap in GAP_VALUES:
        # Option A setup: hold mu_0 fixed and move mu_1 via the gap.
        mu1 = MU0 - gap

        for delta in DELTA_VALUES:
            outcomes = []

            for seed in range(N_RUNS):
                result = run_episode(
                    mu_0=MU0,
                    mu_1=mu1,
                    delta=delta,
                    horizon=HORIZON,
                    seed=seed,
                )
                outcomes.append(result.outcome)

            counts = Counter(outcomes)

            # Shares are easier to compare than raw counts across settings.
            optimal_share = counts.get("optimal_convergence", 0) / N_RUNS
            ambiguous_share = counts.get("ambiguous", 0) / N_RUNS
            lock_in_share = counts.get("lock_in", 0) / N_RUNS

            print(
                f"{delta:>4.2f} | {gap:>4.2f} | "
                f"{optimal_share:>7.3f} | {ambiguous_share:>9.3f} | {lock_in_share:>7.3f}"
            )


if __name__ == "__main__":
    main()