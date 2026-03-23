from dataclasses import dataclass
import numpy as np


@dataclass
class BernoulliBandit:
    """Two-armed Bernoulli environment with fixed arm means."""
    mu_0: float
    mu_1: float
    rng: np.random.Generator

    def pull(self, arm: int) -> int:
        """Draw a binary reward from the chosen arm."""
        if arm == 0:
            return int(self.rng.random() < self.mu_0)
        if arm == 1:
            return int(self.rng.random() < self.mu_1)
        raise ValueError(f"Invalid arm index: {arm}")

    @property
    def optimal_arm(self) -> int:
        """Return the arm with higher expected reward (ties go to arm 1)."""
        return 0 if self.mu_0 > self.mu_1 else 1
    