import numpy as np


def apply_exposure_bias(
    baseline_probs: np.ndarray,
    delta: float,
    inferior_arm: int = 1
) -> np.ndarray:
    """
    Distort baseline action probabilities toward a designated arm.

    Parameters
    ----------
    baseline_probs : np.ndarray
        Length-2 vector of baseline sampling probabilities.
    delta : float
        Exposure distortion parameter.
    inferior_arm : int
        The arm receiving the exposure boost.

    Returns
    -------
    np.ndarray
        Distorted probability vector summing to 1.

    Notes
    -----
    For two arms, we add `delta` to `inferior_arm` (clipped at 1.0) and
    assign the remaining mass to the other arm.
    """
    if len(baseline_probs) != 2:
        raise ValueError("This function currently supports exactly two arms.")

    if not (0.0 <= delta < 1.0):
        raise ValueError("delta must lie in [0, 1).")

    # Shift probability mass toward the designated arm, then renormalize.
    distorted = baseline_probs.copy()
    distorted[inferior_arm] = min(distorted[inferior_arm] + delta, 1.0)
    distorted[1 - inferior_arm] = 1.0 - distorted[inferior_arm]

    if np.any(distorted < 0) or not np.isclose(distorted.sum(), 1.0):
        raise ValueError(f"Invalid distorted probabilities: {distorted}")

    return distorted