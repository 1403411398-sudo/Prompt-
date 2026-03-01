"""
Random Search Optimizer - Randomly samples prompts and evaluates them.
Now with parallel evaluation: evaluates up to 8 prompts simultaneously.
"""
import random
import time
import logging
from typing import Callable

from backend.optimizers.base import BaseOptimizer, OptimizationResult, OptimizationHistory
from backend.prompt_generator import mutate_prompt

logger = logging.getLogger(__name__)


class RandomSearchOptimizer(BaseOptimizer):
    """Random Search: evaluate random prompts and keep the best.
    Uses parallel batch evaluation (8 prompts at a time) for speed.
    """

    @property
    def name(self) -> str:
        return "random_search"

    def optimize(self, initial_prompts: list[str]) -> OptimizationHistory:
        self.history = OptimizationHistory(task_type=self.task_type, algorithm=self.name)
        prompts = list(initial_prompts)
        iteration = 0

        while iteration < self.max_iterations:
            # Collect a batch of up to 8 prompts to evaluate in parallel
            batch = []
            while len(batch) < self.PARALLEL_BATCH_SIZE and iteration + len(batch) < self.max_iterations:
                idx = iteration + len(batch)
                if idx < len(prompts):
                    batch.append(prompts[idx])
                else:
                    # Generate new variants by mutating the best so far
                    base = self.history.best_result.prompt if self.history.best_result else random.choice(prompts)
                    batch.append(mutate_prompt(base, mutation_rate=0.5))

            if not batch:
                break

            batch_start = iteration + 1
            logger.info(
                f"[Random Search] ────── Parallel Batch: iterations "
                f"{batch_start}–{batch_start + len(batch) - 1} / "
                f"{self.max_iterations} ({len(batch)} prompts) ──────"
            )
            for i, p in enumerate(batch):
                preview = p[:80] + '...' if len(p) > 80 else p
                logger.info(f"  Prompt [{batch_start + i}]: {preview}")

            # Evaluate the entire batch in parallel (up to 8 concurrent)
            self._evaluate_batch_parallel(batch, start_iteration=batch_start)

            iteration += len(batch)

        self.history.end_time = time.time()
        return self.history
