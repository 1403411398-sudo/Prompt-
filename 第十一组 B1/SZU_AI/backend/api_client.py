"""
API Client for Alibaba Bailian (DashScope) - Multi-Model Support
Supports: Qwen3.5-35B-A3B, DeepSeek-V3.2, Kimi-K2.5
All models accessed via DashScope's OpenAI-compatible endpoint.
"""
import os
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

API_KEY = os.environ.get("DASHSCOPE_API_KEY", "sk-4636e9d0643c4c63a0687d01e4dfcc55")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ──────────────────────── Model Configurations ────────────────────────

MODEL_CONFIGS = {
    "qwen3.5-35b-a3b": {
        "model_id": "qwen3.5-35b-a3b",
        "display_name": "Qwen3.5-35B-A3B",
        "provider": "通义千问",
        "description": "35B总参数/3B激活，混合架构，支持思考模式",
        "color": "#6366f1",
    },
    "deepseek-v3": {
        "model_id": "deepseek-v3",
        "display_name": "DeepSeek-V3",
        "provider": "DeepSeek",
        "description": "671B MoE架构，高性能推理与代码生成",
        "color": "#06b6d4",
    },
    "kimi-k2": {
        "model_id": "kimi-k2",
        "display_name": "Kimi-K2",
        "provider": "Moonshot AI",
        "description": "原生多模态模型，智能体协作技术",
        "color": "#f59e0b",
    },
}

DEFAULT_MODEL = "qwen3.5-35b-a3b"


def get_available_models() -> list[dict]:
    """Return list of available model configurations."""
    models = []
    for key, cfg in MODEL_CONFIGS.items():
        models.append({
            "id": key,
            "model_id": cfg["model_id"],
            "display_name": cfg["display_name"],
            "provider": cfg["provider"],
            "description": cfg["description"],
            "color": cfg["color"],
            "available": bool(API_KEY),
        })
    return models


def get_client() -> OpenAI:
    """Get a synchronous OpenAI-compatible client for DashScope."""
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def _resolve_model(model: str) -> str:
    """Resolve user-facing model name to DashScope model ID."""
    if model in MODEL_CONFIGS:
        return MODEL_CONFIGS[model]["model_id"]
    return model


def call_llm(prompt: str, user_input: str, model: str = DEFAULT_MODEL,
             temperature: float = 0.3, max_tokens: int = 1024) -> str:
    """
    Call the LLM with a system prompt and user input.
    Returns the assistant's response text.
    """
    client = get_client()
    resolved_model = _resolve_model(model)
    try:
        response = client.chat.completions.create(
            model=resolved_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"LLM API call failed (model={resolved_model}): {e}")
        raise


def call_llm_batch(prompt: str, inputs: list[str], model: str = DEFAULT_MODEL,
                   temperature: float = 0.3, max_tokens: int = 1024) -> list[str]:
    """
    Call the LLM for a batch of inputs with the same system prompt.
    Uses concurrent execution for speed.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _call_single(user_input):
        return call_llm(prompt, user_input, model, temperature, max_tokens)

    results = [None] * len(inputs)
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_idx = {executor.submit(_call_single, inp): i for i, inp in enumerate(inputs)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"Batch call failed for input {idx}: {e}")
                results[idx] = ""
    return results


def evaluate_prompts_parallel(
    prompts: list[str],
    evaluate_fn,
    max_parallel: int = 8,
) -> list[tuple[str, float, dict, list[str]]]:
    """
    Evaluate multiple prompts in parallel, up to `max_parallel` at a time.
    Each prompt is evaluated using the provided evaluate_fn, which itself
    may use call_llm_batch for internal parallelism.

    Args:
        prompts: List of prompt strings to evaluate.
        evaluate_fn: Function(prompt) -> (score, metrics, predictions).
        max_parallel: Maximum number of prompts to evaluate concurrently (default 8).

    Returns:
        List of (prompt, score, metrics, predictions) tuples in the same order as input.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _eval_single(prompt):
        try:
            score, metrics, predictions = evaluate_fn(prompt)
            return (prompt, score, metrics, predictions)
        except Exception as e:
            logger.error(f"Parallel prompt evaluation failed: {e}")
            return (prompt, 0.0, {}, [])

    results = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=min(max_parallel, len(prompts))) as executor:
        future_to_idx = {
            executor.submit(_eval_single, p): i
            for i, p in enumerate(prompts)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"Parallel eval future failed for prompt {idx}: {e}")
                results[idx] = (prompts[idx], 0.0, {}, [])

    return results
