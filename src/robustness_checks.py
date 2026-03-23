from dataclasses import dataclass
import numpy as np

from src.bandit import BernoulliBandit
from src.exposure import apply_exposure_bias
from src.metrics import classify_outcome
from src.simulation import SimulationResult


@dataclass
class UCB1:
    counts: np.ndarray
    reward_sums: np.ndarray
    rng: np.random.Generator

    @classmethod
    def with_zero_state(cls, rng: np.random.Generator, n_arms: int = 2) -> "UCB1":
        return cls(
            counts=np.zeros(n_arms, dtype=int),
            reward_sums=np.zeros(n_arms, dtype=float),
            rng=rng,
        )

    def baseline_probabilities(self) -> np.ndarray:
        """Deterministic UCB1 policy as a probability vector (ties split uniformly)."""
        n_arms = len(self.counts)
        probs = np.zeros(n_arms, dtype=float)
        total_pulls = int(self.counts.sum())

        # Force initial coverage of unpulled arms.
        unpulled = np.where(self.counts == 0)[0]
        if len(unpulled) > 0:
            probs[unpulled] = 1.0 / len(unpulled)
            return probs

        means = self.reward_sums / self.counts
        bonus = np.sqrt(2.0 * np.log(max(1, total_pulls)) / self.counts)
        ucb_values = means + bonus

        best = np.where(np.isclose(ucb_values, np.max(ucb_values)))[0]
        probs[best] = 1.0 / len(best)
        return probs

    def choose_arm(self, probs: np.ndarray) -> int:
        return int(self.rng.choice(np.arange(len(probs)), p=probs))

    def update(self, arm: int, reward: int) -> None:
        self.counts[arm] += 1
        self.reward_sums[arm] += reward


@dataclass
class EpsilonGreedy:
    epsilon: float
    counts: np.ndarray
    reward_sums: np.ndarray
    rng: np.random.Generator

    @classmethod
    def with_zero_state(
        cls,
        rng: np.random.Generator,
        epsilon: float = 0.1,
        n_arms: int = 2,
    ) -> "EpsilonGreedy":
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must lie in [0, 1].")
        return cls(
            epsilon=epsilon,
            counts=np.zeros(n_arms, dtype=int),
            reward_sums=np.zeros(n_arms, dtype=float),
            rng=rng,
        )

    def baseline_probabilities(self) -> np.ndarray:
        """Epsilon-greedy probability vector with tie handling."""
        n_arms = len(self.counts)
        probs = np.full(n_arms, self.epsilon / n_arms, dtype=float)

        # If some arms are unpulled, treat them as greedy candidates.
        unpulled = np.where(self.counts == 0)[0]
        if len(unpulled) > 0:
            greedy_set = unpulled
        else:
            means = self.reward_sums / self.counts
            greedy_set = np.where(np.isclose(means, np.max(means)))[0]

        probs[greedy_set] += (1.0 - self.epsilon) / len(greedy_set)
        return probs

    def choose_arm(self, probs: np.ndarray) -> int:
        return int(self.rng.choice(np.arange(len(probs)), p=probs))

    def update(self, arm: int, reward: int) -> None:
        self.counts[arm] += 1
        self.reward_sums[arm] += reward


def run_episode_ucb1(
    mu_0: float,
    mu_1: float,
    delta: float,
    horizon: int,
    seed: int,
    inferior_arm: int = 1,
) -> SimulationResult:
    rng = np.random.default_rng(seed)
    bandit = BernoulliBandit(mu_0=mu_0, mu_1=mu_1, rng=rng)
    learner = UCB1.with_zero_state(rng=rng, n_arms=2)

    choices = np.zeros(horizon, dtype=int)
    rewards = np.zeros(horizon, dtype=int)
    baseline_prob_history = np.zeros((horizon, 2), dtype=float)
    distorted_prob_history = np.zeros((horizon, 2), dtype=float)

    for t in range(horizon):
        baseline_probs = learner.baseline_probabilities()
        distorted_probs = apply_exposure_bias(
            baseline_probs=baseline_probs,
            delta=delta,
            inferior_arm=inferior_arm,
        )
        arm = learner.choose_arm(distorted_probs)
        reward = bandit.pull(arm)
        learner.update(arm, reward)

        choices[t] = arm
        rewards[t] = reward
        baseline_prob_history[t] = baseline_probs
        distorted_prob_history[t] = distorted_probs

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


def run_episode_epsilon_greedy(
    mu_0: float,
    mu_1: float,
    delta: float,
    horizon: int,
    seed: int,
    epsilon: float = 0.1,
    inferior_arm: int = 1,
) -> SimulationResult:
    rng = np.random.default_rng(seed)
    bandit = BernoulliBandit(mu_0=mu_0, mu_1=mu_1, rng=rng)
    learner = EpsilonGreedy.with_zero_state(rng=rng, epsilon=epsilon, n_arms=2)

    choices = np.zeros(horizon, dtype=int)
    rewards = np.zeros(horizon, dtype=int)
    baseline_prob_history = np.zeros((horizon, 2), dtype=float)
    distorted_prob_history = np.zeros((horizon, 2), dtype=float)

    for t in range(horizon):
        baseline_probs = learner.baseline_probabilities()
        distorted_probs = apply_exposure_bias(
            baseline_probs=baseline_probs,
            delta=delta,
            inferior_arm=inferior_arm,
        )
        arm = learner.choose_arm(distorted_probs)
        reward = bandit.pull(arm)
        learner.update(arm, reward)

        choices[t] = arm
        rewards[t] = reward
        baseline_prob_history[t] = baseline_probs
        distorted_prob_history[t] = distorted_probs

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