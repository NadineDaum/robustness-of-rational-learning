from dataclasses import dataclass
import numpy as np


@dataclass
class ThompsonSampling:
    """
    Beta-Bernoulli Thompson Sampling for a two-armed bandit.

    The posterior for each arm is modeled as Beta(alpha, beta), updated with
    binary rewards. This class exposes helper methods for sampling action
    preferences, choosing an arm, and updating posteriors.
    """
    alpha: np.ndarray
    beta: np.ndarray
    rng: np.random.Generator

    @classmethod
    def with_uniform_priors(cls, rng: np.random.Generator) -> "ThompsonSampling":
        """Create a learner with Beta(1, 1) priors for both arms."""
        return cls(
            alpha=np.array([1.0, 1.0], dtype=float),
            beta=np.array([1.0, 1.0], dtype=float),
            rng=rng,
        )

    def sample_preferences(self) -> np.ndarray:
        """
        Estimate action probabilities via repeated posterior sampling.

        We draw multiple samples from each arm's posterior and compute the
        empirical probability that each arm wins the sample-wise argmax.
        """
        # Monte Carlo estimate of P(arm i is best | current posterior).
        draws = self.rng.beta(self.alpha, self.beta, size=(300, 2))
        chosen = np.argmax(draws, axis=1)
        prob_arm_0 = np.mean(chosen == 0)
        prob_arm_1 = 1.0 - prob_arm_0
        return np.array([prob_arm_0, prob_arm_1])

    def choose_arm(self, probs: np.ndarray) -> int:
        """Choose an arm according to supplied probabilities."""
        return int(self.rng.choice([0, 1], p=probs))

    def update(self, arm: int, reward: int) -> None:
        """Update Beta posterior for the selected arm using a Bernoulli reward."""
        self.alpha[arm] += reward
        self.beta[arm] += (1 - reward)
    