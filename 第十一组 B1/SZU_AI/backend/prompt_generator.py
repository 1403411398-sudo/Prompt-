"""
Prompt Generator - Generates prompt variants using multiple strategies:
  1. Template-based generation
  2. Keyword combination
  3. Semantic expansion
  4. Mutation (for genetic algorithm)
"""
import random
import itertools
from typing import Optional

# ──────────────────────── Classification Templates ────────────────────────
CLASSIFICATION_TEMPLATES = [
    "请将以下新闻文本分类到最合适的类别中。可选类别：{categories}。只输出类别名称。",
    "你是一个专业的新闻分类器。请分析以下文本并判断它属于哪个类别：{categories}。请只回答类别名，不要任何解释。",
    "阅读以下文本，判断其主题类别。类别列表：{categories}。请直接输出类别，不要附加内容。",
    "作为新闻编辑，请将下面的文本归入对应的类别。可用类别有：{categories}。仅输出一个类别名称。",
    "请仔细阅读以下内容，确定它最可能属于以下哪个新闻类别：{categories}。请只返回分类结果。",
    "你的任务是对新闻进行精确分类。请从{categories}中选择最匹配的类别。只输出类别名称，不要解释。",
    "分析以下新闻文本的主题和关键信息，将其归类为{categories}中的一个。只输出类别。",
    "请扮演资深新闻编辑，对以下文本进行分类。候选类别：{categories}。回复格式：仅输出类别名称。",
]

# ──────────────────────── Summarization Templates ────────────────────────
SUMMARIZATION_TEMPLATES = [
    "请对以下文本进行摘要，用{length}概括要点。保持客观准确。",
    "你是一个专业的文本摘要专家。请将以下内容精炼为{length}的摘要，保留关键信息。",
    "阅读以下文本，生成一个简洁的摘要。要求：{length}以内，涵盖核心观点。",
    "请提取以下文本的关键信息，生成{length}左右的摘要。要点完整，语言精练。",
    "作为信息分析师，请将以下长文压缩为{length}的摘要。保留核心事实和数据。",
    "请对以下文本进行概括总结。要求简明扼要，控制在{length}以内，突出重点。",
    "你的任务是提取文本核心内容并生成摘要。字数限制：{length}。包含关键事实和结论。",
    "请从专业角度对以下文本进行摘要提炼。{length}以内，确保信息完整性。",
]

# ──────────────────────── Translation Templates ────────────────────────
TRANSLATION_TEMPLATES = [
    "请将以下中文文本翻译成英文。要求翻译{style}。",
    "你是一个专业翻译，请将以下中文译为英文。翻译要求：{style}。",
    "将以下中文内容翻译为地道的英文。风格要求：{style}。只输出翻译结果。",
    "作为资深中英翻译，请翻译以下文本。翻译标准：{style}。",
    "请将下面的中文{style}地翻译成英文。只输出译文，不要附加说明。",
    "你的任务是高质量中英翻译。请翻译以下文本，要求{style}，语言流畅。",
    "请对以下中文文本进行英文翻译。翻译原则：{style}。直接输出译文。",
    "作为语言专家，请将以下中文翻译为英文。注意{style}。",
]

# ──────────────────────── Keywords / Attributes ────────────────────────
ROLE_KEYWORDS = [
    "专业的", "资深的", "经验丰富的", "权威的", "精通此领域的",
    "学术背景的", "一丝不苟的", "高效的",
]

STYLE_KEYWORDS_CLS = [
    "仔细分析后", "快速判断", "基于关键词", "从语义角度",
    "根据主题和内容", "综合考虑上下文",
]

CONSTRAINT_KEYWORDS = [
    "只输出类别名称", "不要任何解释", "直接给出答案",
    "请只返回结果", "不要附加内容",
    "回答格式：类别名", "一个词回答",
]

SUMMARY_LENGTH_KEYWORDS = [
    "2-3句话", "50字左右", "100字以内", "3句话以内", "一段话",
]

TRANSLATION_STYLE_KEYWORDS = [
    "准确、流畅、自然", "信达雅", "忠实原文且地道自然",
    "专业准确", "简洁明了", "学术风格",
    "口语化且自然", "保持原文风格",
]

COT_KEYWORDS = [
    "", "请先分析文本内容，然后给出答案。",
    "让我们一步步思考。", "请先列出关键信息，再给出结论。",
]


def generate_classification_prompts(categories: list[str], n: int = 10) -> list[str]:
    """Generate diverse classification prompt variants."""
    cat_str = "、".join(categories)
    prompts = []

    # Template-based
    for tpl in CLASSIFICATION_TEMPLATES:
        prompts.append(tpl.format(categories=cat_str))

    # Keyword combination
    for _ in range(n):
        role = random.choice(ROLE_KEYWORDS)
        style = random.choice(STYLE_KEYWORDS_CLS)
        constraint = random.choice(CONSTRAINT_KEYWORDS)
        cot = random.choice(COT_KEYWORDS)
        prompt = f"你是一个{role}新闻分类助手。请{style}，将以下文本分类到：{cat_str}。{constraint}。{cot}"
        prompts.append(prompt.strip())

    return list(set(prompts))[:n + len(CLASSIFICATION_TEMPLATES)]


def generate_summarization_prompts(n: int = 10) -> list[str]:
    """Generate diverse summarization prompt variants."""
    prompts = []

    for tpl in SUMMARIZATION_TEMPLATES:
        length = random.choice(SUMMARY_LENGTH_KEYWORDS)
        prompts.append(tpl.format(length=length))

    for _ in range(n):
        role = random.choice(ROLE_KEYWORDS)
        length = random.choice(SUMMARY_LENGTH_KEYWORDS)
        cot = random.choice(COT_KEYWORDS)
        prompt = f"你是一个{role}文本摘要专家。请将以下文本概括为{length}的摘要。保留关键信息和核心观点。{cot}"
        prompts.append(prompt.strip())

    return list(set(prompts))[:n + len(SUMMARIZATION_TEMPLATES)]


def generate_translation_prompts(n: int = 10) -> list[str]:
    """Generate diverse translation prompt variants."""
    prompts = []

    for tpl in TRANSLATION_TEMPLATES:
        style = random.choice(TRANSLATION_STYLE_KEYWORDS)
        prompts.append(tpl.format(style=style))

    for _ in range(n):
        role = random.choice(ROLE_KEYWORDS)
        style = random.choice(TRANSLATION_STYLE_KEYWORDS)
        cot = random.choice(COT_KEYWORDS)
        prompt = f"你是一个{role}中英翻译专家。请将以下中文翻译为英文。要求{style}。只输出译文。{cot}"
        prompts.append(prompt.strip())

    return list(set(prompts))[:n + len(TRANSLATION_TEMPLATES)]


def mutate_prompt(prompt: str, mutation_rate: float = 0.3) -> str:
    """
    Mutate a prompt by randomly replacing/inserting/deleting keywords.
    Used by genetic algorithm.
    """
    all_keywords = (ROLE_KEYWORDS + STYLE_KEYWORDS_CLS + CONSTRAINT_KEYWORDS +
                    COT_KEYWORDS + SUMMARY_LENGTH_KEYWORDS + TRANSLATION_STYLE_KEYWORDS)

    if random.random() < mutation_rate:
        # Insert a random keyword
        kw = random.choice(all_keywords)
        if kw and kw not in prompt:
            words = prompt.split("。")
            insert_pos = random.randint(0, len(words) - 1)
            words.insert(insert_pos, kw)
            prompt = "。".join(words)

    if random.random() < mutation_rate:
        # Replace a keyword
        for kw_list in [ROLE_KEYWORDS, STYLE_KEYWORDS_CLS, CONSTRAINT_KEYWORDS]:
            for kw in kw_list:
                if kw in prompt:
                    replacement = random.choice(kw_list)
                    prompt = prompt.replace(kw, replacement, 1)
                    break

    return prompt


def crossover_prompts(prompt1: str, prompt2: str) -> str:
    """
    Crossover two prompts by combining parts of each.
    Used by genetic algorithm.
    """
    parts1 = prompt1.split("。")
    parts2 = prompt2.split("。")

    # Take random parts from each
    result_parts = []
    max_len = max(len(parts1), len(parts2))
    for i in range(max_len):
        if random.random() < 0.5 and i < len(parts1):
            result_parts.append(parts1[i])
        elif i < len(parts2):
            result_parts.append(parts2[i])
        elif i < len(parts1):
            result_parts.append(parts1[i])

    result = "。".join(result_parts)
    # Clean up
    result = result.replace("。。", "。").strip()
    if not result.endswith("。"):
        result += "。"
    return result


def generate_prompts_for_task(task_type: str, n: int = 10, categories: Optional[list[str]] = None) -> list[str]:
    """Generate prompts based on task type."""
    if task_type == "classification":
        cats = categories or ["科技", "体育", "财经", "健康", "娱乐"]
        return generate_classification_prompts(cats, n)
    elif task_type == "summarization":
        return generate_summarization_prompts(n)
    elif task_type == "translation":
        return generate_translation_prompts(n)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
