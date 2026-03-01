"""
Evaluator - Computes metrics for different task types:
  - Classification: Accuracy
  - Summarization: ROUGE-1, ROUGE-2, ROUGE-L
  - Translation: BLEU
  - LLM-as-Judge: uses LLM to score outputs
"""
import re
import logging
import math
from collections import Counter
from typing import Optional

import jieba

from backend.api_client import call_llm

logger = logging.getLogger(__name__)


# ──────────────────────── Tokenization ────────────────────────

def tokenize_chinese(text: str) -> list[str]:
    """Tokenize Chinese text using jieba."""
    return list(jieba.cut(text))


def tokenize_text(text: str) -> list[str]:
    """Tokenize text - use jieba for Chinese, simple split for English."""
    if any('\u4e00' <= ch <= '\u9fff' for ch in text):
        return tokenize_chinese(text)
    return text.lower().split()


# ──────────────────────── Accuracy (Classification) ────────────────────────

def compute_accuracy(predictions: list[str], labels: list[str]) -> float:
    """Compute classification accuracy with fuzzy matching."""
    if not predictions or not labels:
        return 0.0

    correct = 0
    for pred, label in zip(predictions, labels):
        pred_clean = pred.strip().replace(" ", "")
        label_clean = label.strip().replace(" ", "")
        if label_clean in pred_clean or pred_clean == label_clean:
            correct += 1

    return correct / len(labels)


# ──────────────────────── BLEU (Translation) ────────────────────────

def compute_ngrams(tokens: list[str], n: int) -> Counter:
    """Compute n-gram counts."""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def compute_bleu(prediction: str, reference: str, max_n: int = 4) -> float:
    """Compute BLEU score for a single prediction-reference pair."""
    pred_tokens = tokenize_text(prediction)
    ref_tokens = tokenize_text(reference)

    if len(pred_tokens) == 0:
        return 0.0

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1)))

    # N-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = compute_ngrams(pred_tokens, n)
        ref_ngrams = compute_ngrams(ref_tokens, n)

        clipped = sum(min(pred_ngrams[ng], ref_ngrams[ng]) for ng in pred_ngrams)
        total = sum(pred_ngrams.values())

        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped / total)

    # Geometric mean of precisions
    log_precisions = []
    for p in precisions:
        if p > 0:
            log_precisions.append(math.log(p))
        else:
            return 0.0

    avg_log = sum(log_precisions) / len(log_precisions)
    return bp * math.exp(avg_log)


def compute_bleu_batch(predictions: list[str], references: list[str]) -> float:
    """Compute average BLEU score for a batch."""
    if not predictions or not references:
        return 0.0

    scores = [compute_bleu(p, r) for p, r in zip(predictions, references)]
    return sum(scores) / len(scores)


# ──────────────────────── ROUGE (Summarization) ────────────────────────

def compute_rouge_n(prediction: str, reference: str, n: int = 1) -> dict:
    """Compute ROUGE-N score."""
    pred_tokens = tokenize_text(prediction)
    ref_tokens = tokenize_text(reference)

    pred_ngrams = compute_ngrams(pred_tokens, n)
    ref_ngrams = compute_ngrams(ref_tokens, n)

    overlap = sum(min(pred_ngrams[ng], ref_ngrams[ng]) for ng in ref_ngrams)
    ref_count = sum(ref_ngrams.values())
    pred_count = sum(pred_ngrams.values())

    precision = overlap / pred_count if pred_count > 0 else 0.0
    recall = overlap / ref_count if ref_count > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def lcs_length(x: list, y: list) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_rouge_l(prediction: str, reference: str) -> dict:
    """Compute ROUGE-L score based on LCS."""
    pred_tokens = tokenize_text(prediction)
    ref_tokens = tokenize_text(reference)

    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = lcs_length(pred_tokens, ref_tokens)

    precision = lcs / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = lcs / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_rouge_batch(predictions: list[str], references: list[str]) -> dict:
    """Compute average ROUGE scores for a batch."""
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for pred, ref in zip(predictions, references):
        rouge1_scores.append(compute_rouge_n(pred, ref, 1)["f1"])
        rouge2_scores.append(compute_rouge_n(pred, ref, 2)["f1"])
        rougeL_scores.append(compute_rouge_l(pred, ref)["f1"])

    return {
        "rouge1": sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
        "rouge2": sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0,
        "rougeL": sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0,
    }


# ──────────────────────── LLM-as-Judge ────────────────────────

LLM_JUDGE_PROMPT = """你是一个专业的评估专家。请根据以下标准对给定的输出进行评分。

任务类型：{task_type}
评估标准：
1. 准确性（0-10分）：输出是否准确完成了任务
2. 完整性（0-10分）：输出是否包含了所有关键信息
3. 语言质量（0-10分）：输出的语言是否流畅自然

请以JSON格式返回评分，格式如下：
{{"accuracy": <分数>, "completeness": <分数>, "language_quality": <分数>, "overall": <总分0-10>}}

任务描述：{task_desc}
输入：{input_text}
{ref_text}
模型输出：{output_text}

请直接返回JSON格式的评分，不要附加任何其他内容。"""


def llm_judge(task_type: str, input_text: str, output_text: str,
              reference: Optional[str] = None) -> dict:
    """Use LLM as judge to evaluate output quality."""
    task_desc_map = {
        "classification": "将文本分类到正确的类别",
        "summarization": "对文本进行摘要",
        "translation": "将中文翻译为英文",
    }

    ref_text = f"参考答案：{reference}" if reference else "（无参考答案）"

    judge_prompt = LLM_JUDGE_PROMPT.format(
        task_type=task_type,
        task_desc=task_desc_map.get(task_type, task_type),
        input_text=input_text[:500],
        ref_text=ref_text,
        output_text=output_text[:500],
    )

    try:
        response = call_llm(
            "你是一个严格的评估专家，请以JSON格式返回评分。",
            judge_prompt,
            temperature=0.1,
        )

        # Parse JSON from response
        import json
        # Try to extract JSON from the response
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            scores = json.loads(json_match.group())
            return scores
        else:
            logger.warning(f"Could not parse LLM judge response: {response}")
            return {"accuracy": 5, "completeness": 5, "language_quality": 5, "overall": 5}
    except Exception as e:
        logger.error(f"LLM judge failed: {e}")
        return {"accuracy": 5, "completeness": 5, "language_quality": 5, "overall": 5}


def llm_judge_batch(task_type: str, inputs: list[str], outputs: list[str],
                    references: Optional[list[str]] = None) -> float:
    """Compute average LLM judge score for a batch. Returns normalized score 0-1."""
    scores = []
    for i, (inp, out) in enumerate(zip(inputs, outputs)):
        ref = references[i] if references else None
        result = llm_judge(task_type, inp, out, ref)
        overall = result.get("overall", 5) / 10.0
        scores.append(overall)
    return sum(scores) / len(scores) if scores else 0.0


# ──────────────────────── Unified Evaluation ────────────────────────

def evaluate(task_type: str, predictions: list[str], references: list[str],
             inputs: Optional[list[str]] = None,
             use_llm_judge: bool = False) -> dict:
    """
    Unified evaluation function.
    Returns a dict with the primary score and detailed metrics.
    """
    result = {"task_type": task_type}

    if task_type == "classification":
        acc = compute_accuracy(predictions, references)
        result["accuracy"] = round(acc, 4)
        result["primary_score"] = round(acc, 4)
        result["primary_metric"] = "accuracy"

    elif task_type == "summarization":
        rouge_scores = compute_rouge_batch(predictions, references)
        result["rouge1"] = round(rouge_scores["rouge1"], 4)
        result["rouge2"] = round(rouge_scores["rouge2"], 4)
        result["rougeL"] = round(rouge_scores["rougeL"], 4)
        result["primary_score"] = round(rouge_scores["rougeL"], 4)
        result["primary_metric"] = "rougeL"

    elif task_type == "translation":
        bleu = compute_bleu_batch(predictions, references)
        result["bleu"] = round(bleu, 4)
        result["primary_score"] = round(bleu, 4)
        result["primary_metric"] = "bleu"

    else:
        raise ValueError(f"Unknown task type: {task_type}")

    # Optionally use LLM as judge
    if use_llm_judge and inputs:
        llm_score = llm_judge_batch(task_type, inputs, predictions, references)
        result["llm_judge_score"] = round(llm_score, 4)

    return result
