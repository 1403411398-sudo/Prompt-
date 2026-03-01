"""
Genetic Algorithm Optimizer - Uses selection, crossover, and mutation to evolve prompts.
Now with parallel evaluation: evaluates the entire population simultaneously (up to 8).
"""
import random
import time
import logging
from typing import Callable

from backend.optimizers.base import BaseOptimizer, OptimizationResult, OptimizationHistory
from backend.prompt_generator import mutate_prompt, crossover_prompts

logger = logging.getLogger(__name__)


class GeneticOptimizer(BaseOptimizer):
    """Genetic Algorithm: evolve a population of prompts.
    Uses parallel batch evaluation for the entire population each generation.
    """

    def __init__(self, task_type: str, evaluate_fn: Callable,
                 max_iterations: int = 20,
                 population_size: int = 6,
                 elite_count: int = 2,
                 mutation_rate: float = 0.3):
        super().__init__(task_type, evaluate_fn, max_iterations)
        self.population_size = population_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate

    @property
    def name(self) -> str:
        return "genetic"

    def _select_parents(self, population_scores: list[tuple[str, float]]) -> list[str]:
        """Tournament selection."""
        selected = []
        for _ in range(2):
            tournament = random.sample(population_scores, min(3, len(population_scores)))
            winner = max(tournament, key=lambda x: x[1])
            selected.append(winner[0])
        return selected

    def optimize(self, initial_prompts: list[str]) -> OptimizationHistory:
        self.history = OptimizationHistory(task_type=self.task_type, algorithm=self.name)

        # Initialize population
        population = list(initial_prompts[:self.population_size])
        while len(population) < self.population_size:
            base = random.choice(initial_prompts)
            population.append(mutate_prompt(base, 0.5))

        generation = 0
        iteration = 0

        while iteration < self.max_iterations:
            generation += 1

            # Trim population to fit within remaining iterations
            remaining = self.max_iterations - iteration
            batch = population[:remaining]

            logger.info(
                f"[Genetic] ────── Generation {generation}: "
                f"{len(batch)} prompts (parallel) ──────"
            )
            for i, p in enumerate(batch):
                preview = p[:80] + '...' if len(p) > 80 else p
                logger.info(f"  Prompt [{iteration + i + 1}]: {preview}")

            # Evaluate the entire population in parallel (up to 8 concurrent)
            results = self._evaluate_batch_parallel(
                batch, start_iteration=iteration + 1
            )

            # Build population scores from results
            pop_scores = [
                (r.prompt, r.score) for r in results
            ]

            iteration += len(batch)

            if iteration >= self.max_iterations:
                break

            # Sort by score
            pop_scores.sort(key=lambda x: x[1], reverse=True)

            # Elite selection
            new_population = [p for p, _ in pop_scores[:self.elite_count]]

            # Generate offspring
            while len(new_population) < self.population_size:
                parents = self._select_parents(pop_scores)
                child = crossover_prompts(parents[0], parents[1])
                child = mutate_prompt(child, self.mutation_rate)
                new_population.append(child)

            population = new_population

        self.history.end_time = time.time()
        return self.history
