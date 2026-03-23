from collections import Counter

from src.simulation import run_episode


def main() -> None:
    # Single-condition benchmark to see outcome frequencies quickly 
    n_runs = 100
    outcomes = []

    for seed in range(n_runs):
        result = run_episode(
            mu_0=0.6,
            mu_1=0.5,
            delta=0.1,
            horizon=5_000,
            seed=seed,
        )
        outcomes.append(result.outcome)

    counts = Counter(outcomes)

    # Raw counts first, then normalized shares for interpretation 
    print("Outcome counts:")
    for outcome, count in counts.items():
        print(f"{outcome}: {count}")

    print("\nOutcome shares:")
    for outcome, count in counts.items():
        print(f"{outcome}: {count / n_runs:.3f}")


if __name__ == "__main__":
    main()