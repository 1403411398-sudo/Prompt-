"""
Base Optimizer - Abstract interface for all optimizers.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable
import time


@dataclass
class OptimizationResult:
    """Result of a single prompt evaluation."""
    prompt: str
    score: float
    metrics: dict
    iteration: int
    timestamp: float = field(default_factory=time.time)
    predictions: list[str] = field(default_factory=list)


@dataclass
class OptimizationHistory:
    """Full optimization history."""
    task_type: str
    algorithm: str
    results: list[OptimizationResult] = field(default_factory=list)
    best_result: Optional[OptimizationResult] = None
    score_curve: list[float] = field(default_factory=list)
    best_score_curve: list[float] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    keyword_contributions: dict = field(default_factory=dict)

    def add_result(self, result: OptimizationResult):
        self.results.append(result)
        self.score_curve.append(result.score)

        if self.best_result is None or result.score > self.best_result.score:
            self.best_result = result

        self.best_score_curve.append(self.best_result.score)

    def to_dict(self) -> dict:
        return {
            "task_type": self.task_type,
            "algorithm": self.algorithm,
            "total_iterations": len(self.results),
            "best_prompt": self.best_result.prompt if self.best_result else "",
            "best_score": self.best_result.score if self.best_result else 0.0,
            "best_metrics": self.best_result.metrics if self.best_result else {},
            "score_curve": self.score_curve,
            "best_score_curve": self.best_score_curve,
            "all_results": [
                {
                    "prompt": r.prompt,
                    "score": r.score,
                    "metrics": r.metrics,
                    "iteration": r.iteration,
                }
                for r in self.results
            ],
            "duration": (self.end_time or time.time()) - self.start_time,
            "keyword_contributions": self.keyword_contributions,
        }


class BaseOptimizer(ABC):
    """Abstract base class for prompt optimizers."""

    PARALLEL_BATCH_SIZE = 8  # Max prompts to evaluate in parallel

    def __init__(self, task_type: str, evaluate_fn: Callable, max_iterations: int = 20):
        self.task_type = task_type
        self.evaluate_fn = evaluate_fn
        self.max_iterations = max_iterations
        self.history = OptimizationHistory(task_type=task_type, algorithm=self.name)
        self._callback = None

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def optimize(self, initial_prompts: list[str]) -> OptimizationHistory:
        pass

    def set_callback(self, callback: Callable):
        """Set a callback function to be called after each iteration."""
        self._callback = callback

    def _notify(self, result: OptimizationResult):
        """Notify callback of new result."""
        if self._callback:
            self._callback(result)

    def _evaluate_batch_parallel(
        self,
        prompts: list[str],
        start_iteration: int,
    ) -> list[OptimizationResult]:
        """
        Evaluate multiple prompts in parallel (up to PARALLEL_BATCH_SIZE=8).
        Returns a list of OptimizationResult in the same order as input prompts.
        Each result is added to history and notified via callback.
        """
        from backend.api_client import evaluate_prompts_parallel
        import logging

        logger = logging.getLogger(__name__)

        batch_size = min(len(prompts), self.PARALLEL_BATCH_SIZE)
        logger.info(
            f"[{self.name}] ⚡ Parallel evaluation: {len(prompts)} prompts "
            f"(batch_size={batch_size})"
        )

        raw_results = evaluate_prompts_parallel(
            prompts, self.evaluate_fn, max_parallel=batch_size
        )

        opt_results = []
        for i, (prompt, score, metrics, predictions) in enumerate(raw_results):
            iteration = start_iteration + i
            result = OptimizationResult(
                prompt=prompt,
                score=score,
                metrics=metrics,
                iteration=iteration,
                predictions=predictions,
            )
            self.history.add_result(result)
            self._notify(result)
            opt_results.append(result)

            prompt_preview = prompt[:80] + '...' if len(prompt) > 80 else prompt
            best_score = self.history.best_result.score if self.history.best_result else score
            is_best = score >= best_score
            best_marker = " ⭐ NEW BEST!" if is_best and iteration > 1 else ""
            logger.info(
                f"  [{iteration}] Score: {score:.4f} | "
                f"Best: {best_score:.4f}{best_marker}"
            )

        return opt_results
