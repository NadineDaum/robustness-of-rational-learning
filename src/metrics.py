import numpy as np


def tail_choice_share(choice_history: np.ndarray, arm: int, tail_fraction: float = 0.2) -> float:
    """
    Compute how often an arm is chosen in the final portion of a trajectory.

    This focuses measurement on terminal behavior rather than early
    exploration noise.
    """
    if not (0.0 < tail_fraction <= 1.0):
        raise ValueError("tail_fraction must lie in (0, 1].")

    start = int(len(choice_history) * (1.0 - tail_fraction))
    tail = choice_history[start:]
    return float(np.mean(tail == arm))


def classify_outcome(
    choice_history: np.ndarray,
    optimal_arm: int,
    convergence_threshold: float = 0.90,
    tail_fraction: float = 0.2,
) -> str:
    """
    Classify one run into three outcome categories:
    - 'optimal_convergence'
    - 'lock_in'
    - 'ambiguous'

    The rule is threshold-based on tail choice shares. If neither arm exceeds
    the threshold, the run remains ambiguous.
    """
    optimal_share = tail_choice_share(choice_history, optimal_arm, tail_fraction)
    inferior_share = tail_choice_share(choice_history, 1 - optimal_arm, tail_fraction)

    if optimal_share >= convergence_threshold:
        return "optimal_convergence"
    if inferior_share >= convergence_threshold:
        return "lock_in"
    return "ambiguous"