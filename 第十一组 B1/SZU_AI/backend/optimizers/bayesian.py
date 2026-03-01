"""
Bayesian Optimization for Prompt Search.
Uses a simplified Gaussian Process surrogate to model prompt performance
and an acquisition function (Expected Improvement) to select next prompts.

We encode prompts as feature vectors based on keyword presence,
then use the GP surrogate to predict scores for unseen prompts.

Now with parallel evaluation: evaluates up to 8 prompts simultaneously.
"""
import random
import time
import math
import logging
import numpy as np
from typing import Callable

from backend.optimizers.base import BaseOptimizer, OptimizationResult, OptimizationHistory
from backend.prompt_generator import (
    mutate_prompt,
    ROLE_KEYWORDS, STYLE_KEYWORDS_CLS, CONSTRAINT_KEYWORDS,
    COT_KEYWORDS, SUMMARY_LENGTH_KEYWORDS, TRANSLATION_STYLE_KEYWORDS,
)

logger = logging.getLogger(__name__)

ALL_KEYWORDS = (ROLE_KEYWORDS + STYLE_KEYWORDS_CLS + CONSTRAINT_KEYWORDS +
                COT_KEYWORDS + SUMMARY_LENGTH_KEYWORDS + TRANSLATION_STYLE_KEYWORDS)
# Remove empty strings
ALL_KEYWORDS = [kw for kw in ALL_KEYWORDS if kw]


def prompt_to_features(prompt: str) -> np.ndarray:
    """Encode a prompt as a binary feature vector based on keyword presence."""
    features = [1.0 if kw in prompt else 0.0 for kw in ALL_KEYWORDS]
    # Add length features
    features.append(len(prompt) / 200.0)  # normalized length
    features.append(prompt.count("。") / 10.0)  # sentence count
    return np.array(features)


def rbf_kernel(x1: np.ndarray, x2: np.ndarray, length_scale: float = 1.0) -> float:
    """RBF (Gaussian) kernel."""
    diff = x1 - x2
    return math.exp(-0.5 * np.dot(diff, diff) / (length_scale ** 2))


def compute_kernel_matrix(X: np.ndarray, length_scale: float = 1.0) -> np.ndarray:
    """Compute the kernel matrix for a set of feature vectors."""
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = rbf_kernel(X[i], X[j], length_scale)
    return K


class SimplifiedGP:
    """Simplified Gaussian Process for surrogate modeling."""

    def __init__(self, length_scale: float = 1.0, noise: float = 0.1):
        self.length_scale = length_scale
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K_inv = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y
        K = compute_kernel_matrix(X, self.length_scale)
        K += self.noise ** 2 * np.eye(len(X))
        try:
            self.K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            self.K_inv = np.linalg.pinv(K)

    def predict(self, x: np.ndarray) -> tuple[float, float]:
        """Predict mean and std for a single point."""
        if self.X_train is None:
            return 0.5, 1.0

        k_star = np.array([rbf_kernel(x, xi, self.length_scale) for xi in self.X_train])
        k_star_star = rbf_kernel(x, x, self.length_scale)

        mu = k_star @ self.K_inv @ self.y_train
        var = k_star_star - k_star @ self.K_inv @ k_star
        sigma = max(math.sqrt(abs(var)), 1e-6)

        return float(mu), float(sigma)


def expected_improvement(mu: float, sigma: float, best_score: float, xi: float = 0.01) -> float:
    """Compute Expected Improvement acquisition function."""
    if sigma <= 0:
        return 0.0

    z = (mu - best_score - xi) / sigma

    # Approximate normal CDF & PDF
    def norm_cdf(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def norm_pdf(x):
        return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)

    ei = (mu - best_score - xi) * norm_cdf(z) + sigma * norm_pdf(z)
    return max(ei, 0.0)


class BayesianOptimizer(BaseOptimizer):
    """Bayesian Optimization for prompt search.
    Uses parallel evaluation for both exploration and exploitation phases.
    """

    def __init__(self, task_type: str, evaluate_fn: Callable,
                 max_iterations: int = 20,
                 n_initial: int = 5,
                 n_candidates: int = 20):
        super().__init__(task_type, evaluate_fn, max_iterations)
        self.n_initial = n_initial
        self.n_candidates = n_candidates
        self.gp = SimplifiedGP()

    @property
    def name(self) -> str:
        return "bayesian"

    def _generate_candidates(self, observed_prompts: list[str],
                             initial_prompts: list[str]) -> list[str]:
        """Generate candidate prompts for acquisition function evaluation."""
        candidates = []

        # Mutate observed prompts
        for p in observed_prompts:
            for _ in range(2):
                candidates.append(mutate_prompt(p, 0.4))

        # Add from initial pool
        remaining = [p for p in initial_prompts if p not in observed_prompts]
        candidates.extend(remaining[:5])

        # Random mutations of the best
        if observed_prompts:
            best = observed_prompts[0]
            for _ in range(5):
                candidates.append(mutate_prompt(best, 0.6))

        return candidates[:self.n_candidates]

    def _select_top_candidates(self, candidates: list[str],
                               observed_y: list[float],
                               n: int = 8) -> list[str]:
        """Use the GP surrogate to rank candidates by Expected Improvement,
        and return the top-N for parallel evaluation."""
        best_score = max(observed_y) if observed_y else 0.0
        scored = []
        for cand in candidates:
            feat = prompt_to_features(cand)
            mu, sigma = self.gp.predict(feat)
            ei = expected_improvement(mu, sigma, best_score)
            scored.append((cand, ei))

        scored.sort(key=lambda x: x[1], reverse=True)
        # Return the top-N unique candidates
        seen = set()
        top = []
        for cand, ei in scored:
            if cand not in seen:
                seen.add(cand)
                top.append(cand)
            if len(top) >= n:
                break
        return top

    def optimize(self, initial_prompts: list[str]) -> OptimizationHistory:
        self.history = OptimizationHistory(task_type=self.task_type, algorithm=self.name)

        observed_X = []
        observed_y = []
        observed_prompts = []
        iteration = 0

        # Phase 1: Initial random exploration (parallel)
        n_init = min(self.n_initial, self.max_iterations, len(initial_prompts))
        init_prompts = random.sample(initial_prompts, n_init)

        logger.info(
            f"[Bayesian] ────── Phase 1: Parallel Exploration "
            f"({n_init} prompts) ──────"
        )
        for i, p in enumerate(init_prompts):
            preview = p[:80] + '...' if len(p) > 80 else p
            logger.info(f"  Prompt [{i + 1}]: {preview}")

        # Evaluate all initial prompts in parallel
        results = self._evaluate_batch_parallel(
            init_prompts, start_iteration=1
        )

        for result in results:
            observed_X.append(prompt_to_features(result.prompt))
            observed_y.append(result.score)
            observed_prompts.append(result.prompt)

        iteration = n_init

        # Phase 2: Bayesian optimization loop (parallel batches)
        while iteration < self.max_iterations:
            remaining = self.max_iterations - iteration

            if len(observed_X) >= 2:
                # Fit GP on observed data
                X_arr = np.array(observed_X)
                y_arr = np.array(observed_y)
                self.gp.fit(X_arr, y_arr)

                # Generate candidates and rank by Expected Improvement
                candidates = self._generate_candidates(
                    observed_prompts, initial_prompts
                )

                # Select up to 8 (or remaining) top candidates for parallel eval
                batch_size = min(self.PARALLEL_BATCH_SIZE, remaining)
                batch = self._select_top_candidates(
                    candidates, observed_y, n=batch_size
                )
            else:
                batch_size = min(self.PARALLEL_BATCH_SIZE, remaining)
                batch = random.sample(
                    initial_prompts,
                    min(batch_size, len(initial_prompts))
                )

            if not batch:
                break

            logger.info(
                f"[Bayesian] ────── Phase 2: Parallel Batch "
                f"(iterations {iteration + 1}–{iteration + len(batch)} / "
                f"{self.max_iterations}, {len(batch)} prompts) ──────"
            )
            for i, p in enumerate(batch):
                preview = p[:80] + '...' if len(p) > 80 else p
                logger.info(f"  Prompt [{iteration + i + 1}]: {preview}")

            # Evaluate the batch in parallel
            results = self._evaluate_batch_parallel(
                batch, start_iteration=iteration + 1
            )

            for result in results:
                observed_X.append(prompt_to_features(result.prompt))
                observed_y.append(result.score)
                observed_prompts.append(result.prompt)

            iteration += len(batch)

        self.history.end_time = time.time()
        return self.history
