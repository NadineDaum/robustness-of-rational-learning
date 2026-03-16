from src.simulation import run_episode
from src.metrics import tail_choice_share


def main() -> None:
    """
    Run a small number of simulations and inspect tail behavior directly.

    This is a diagnostic script used to understand whether 'ambiguous'
    outcomes are truly unresolved or simply fail to meet a strict
    convergence threshold.
    """
    for seed in range(10):
        # Same parameters, different seeds: isolate stochastic trajectory effects.
        result = run_episode(
            mu_0=0.6,
            mu_1=0.5,
            delta=0.1,
            horizon=2_000,
            seed=seed,
        )

        optimal_tail = tail_choice_share(
            choice_history=result.choices,
            arm=0,
            tail_fraction=0.2,
        )

        inferior_tail = tail_choice_share(
            choice_history=result.choices,
            arm=1,
            tail_fraction=0.2,
        )

        # If both tails are moderate, "ambiguous" is doing its job.
        print(
            f"seed={seed:2d} | outcome={result.outcome:20s} | "
            f"optimal_tail={optimal_tail:.3f} | inferior_tail={inferior_tail:.3f}"
        )


if __name__ == "__main__":
    main()