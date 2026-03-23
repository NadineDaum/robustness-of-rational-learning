from dataclasses import dataclass
import numpy as np

from src.bandit import BernoulliBandit
from src.algorithms import ThompsonSampling
from src.exposure import apply_exposure_bias
from src.metrics import classify_outcome


@dataclass
class SimulationResult:
    """Container for one episode's trajectories and final outcome label."""
    choices: np.ndarray
    rewards: np.ndarray
    baseline_probs: np.ndarray
    distorted_probs: np.ndarray
    outcome: str


def run_episode(
    mu_0: float,
    mu_1: float,
    delta: float,
    horizon: int,
    seed: int,
) -> SimulationResult:
    """
    Run one Thompson Sampling episode under exposure distortion.

    The function records arm choices, rewards, baseline probabilities,
    distorted probabilities, and a final tail-based outcome label.
    """
    rng = np.random.default_rng(seed)

    bandit = BernoulliBandit(mu_0=mu_0, mu_1=mu_1, rng=rng)
    learner = ThompsonSampling.with_uniform_priors(rng=rng)

    choices = np.zeros(horizon, dtype=int)
    rewards = np.zeros(horizon, dtype=int)
    baseline_prob_history = np.zeros((horizon, 2), dtype=float)
    distorted_prob_history = np.zeros((horizon, 2), dtype=float)

    for t in range(horizon):
        # Baseline learner belief over which arm is currently best.
        baseline_probs = learner.sample_preferences()
        
        # External exposure mechanism shifts probabilities toward arm 1.
        distorted_probs = apply_exposure_bias(
            baseline_probs=baseline_probs,
            delta=delta,
            inferior_arm=1,
        )

        arm = learner.choose_arm(distorted_probs)
        reward = bandit.pull(arm)
        learner.update(arm, reward)

        choices[t] = arm
        rewards[t] = reward
        baseline_prob_history[t] = baseline_probs
        distorted_prob_history[t] = distorted_probs

    # Outcome is computed once on the complete trajectory.
    outcome = classify_outcome(
        choice_history=choices,
        optimal_arm=bandit.optimal_arm,
        convergence_threshold=0.90,
        tail_fraction=0.2,
    )

    return SimulationResult(
        choices=choices,
        rewards=rewards,
        baseline_probs=baseline_prob_history,
        distorted_probs=distorted_prob_history,
        outcome=outcome,
    )