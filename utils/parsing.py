from __future__ import annotations

import json
import re
from typing import Any


# ============================================================
# Baseline / CoT 最终表达式解析
# ============================================================


# 匹配完整 Answer 行。
#
# 支持：
#     Answer: (4 + 8) * (6 - 4) = 24
#
# 也支持：
#     answer: ...
#
# [^\r\n]+? 表示只读取当前行，
# 防止把 CoT 中前面的 Steps 一起提取进去。
ANSWER_PATTERN = re.compile(
    r"Answer\s*:\s*"
    r"(?P<expression>[^\r\n]+?)"
    r"\s*=\s*24\b",
    flags=re.IGNORECASE,
)


# 当模型没有输出 Answer: 标签，
# 但输出了：
#
#     (4 + 8) * (6 - 4) = 24
#
# 时使用这个模式。
#
# ^ 和 re.MULTILINE 保证从某一行开头匹配。
FINAL_LINE_PATTERN = re.compile(
    r"^\s*"
    r"(?P<expression>[0-9+\-*/().\s]+?)"
    r"\s*=\s*24\b",
    flags=(
        re.IGNORECASE
        | re.MULTILINE
    ),
)


def extract_final_expression(
    response: str,
) -> str | None:
    """
    从 Baseline 或 CoT 模型输出中提取最终算术表达式。

    支持：

        Answer: (4 + 8) * (6 - 4) = 24

    也支持：

        (4 + 8) * (6 - 4) = 24

    还支持模型只输出：

        (4 + 8) * (6 - 4)

    返回：

        "(4 + 8) * (6 - 4)"

    无法可靠提取时返回 None。
    """
    normalized = _normalize_symbols(response)

    # 优先寻找带 Answer: 的结果。
    #
    # CoT 输出中可能有多个等号，
    # 但最终答案一般在最后，因此取最后一个匹配项。
    answer_matches = list(
        ANSWER_PATTERN.finditer(normalized)
    )

    if answer_matches:
        expression = (
            answer_matches[-1]
            .group("expression")
        )

        expression = _clean_expression(
            expression
        )

        if _looks_like_pure_expression(
            expression
        ):
            return expression

        return None

    # 如果不存在 Answer:，寻找最后一行：
    #
    #     expression = 24
    final_line_matches = list(
        FINAL_LINE_PATTERN.finditer(
            normalized
        )
    )

    if final_line_matches:
        expression = (
            final_line_matches[-1]
            .group("expression")
        )

        expression = _clean_expression(
            expression
        )

        if _looks_like_pure_expression(
            expression
        ):
            return expression

        return None

    # 最后兜底：
    #
    # Baseline prompt 本身以 Answer: 结束，
    # 因此模型可能只补全文本：
    #
    #     (4 + 8) * (6 - 4)
    #
    # 只有整个输出都是纯算术式时才接受，
    # 防止把多行 CoT 步骤误认为最终表达式。
    stripped = normalized.strip()

    if _looks_like_pure_expression(
        stripped
    ):
        return _clean_expression(
            stripped
        )

    return None


def _normalize_symbols(
    text: str,
) -> str:
    """
    把模型可能输出的 Unicode 运算符转换为 verifier 支持的符号。

    例如：

        × -> *
        ÷ -> /
        − -> -
    """
    return (
        text
        .replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("–", "-")
        .replace("＋", "+")
        .replace("／", "/")
        .replace("＊", "*")
    )


def _clean_expression(
    expression: str,
) -> str:
    """
    清理表达式前后的标签和多余空格。
    """
    expression = expression.strip()

    # 某些模型可能重复输出 Answer:。
    expression = re.sub(
        r"^\s*Answer\s*:\s*",
        "",
        expression,
        flags=re.IGNORECASE,
    )

    # 如果末尾仍然带有 = 24，则删除。
    expression = re.sub(
        r"\s*=\s*24\s*$",
        "",
        expression,
        flags=re.IGNORECASE,
    )

    return expression.strip()


def _looks_like_pure_expression(
    text: str,
) -> bool:
    """
    检查字符串是否只包含算术表达式允许的字符。

    这里只负责格式过滤，不负责判断：

        1. 是否使用了正确数字；
        2. 是否每个数字只使用一次；
        3. 结果是否等于 24；
        4. AST 是否合法。

    这些由 verifier 完成。
    """
    if not text:
        return False

    # 不允许多行，避免把 CoT 步骤误识别成一个表达式。
    if "\n" in text or "\r" in text:
        return False

    # 允许：
    #
    #     数字
    #     + - * /
    #     括号
    #     小数点
    #     空格
    #
    # 即使正则允许小数点，
    # 你当前的 verifier 只允许 int literal，
    # 因此 4.0 最后仍会验证失败。
    return (
        re.fullmatch(
            r"[0-9+\-*/().\s]+",
            text,
        )
        is not None
    )


# ============================================================
# ToT 状态评价 JSON 解析
# ============================================================


VALID_STATE_RATINGS = {
    "sure",
    "likely",
    "maybe",
    "impossible",
}

RATING_TO_SCORE = {
    "sure": 1.0,
    "likely": 0.75,
    "maybe": 0.35,
    "impossible": 0.0,
}


def parse_state_scores(
    response: str,
    expected_count: int,
) -> list[float]:
    """
    Parse compact ToT state scores.

    Preferred model output:

        {"scores":[1.0,0.75,0.35,0.0]}

    Older rating output is still accepted as a fallback.
    """
    if expected_count < 0:
        raise ValueError(
            "expected_count must be non-negative"
        )

    if expected_count == 0:
        return []

    cleaned = _remove_code_fence(
        response.strip()
    )

    data = _parse_json_object(cleaned)

    if data is None:
        return _default_scores(
            expected_count
        )

    scores = _parse_scores_array(
        data.get("scores"),
        expected_count,
    )

    if scores is not None:
        return scores

    return [
        RATING_TO_SCORE[rating]
        for rating in parse_state_ratings(
            response,
            expected_count,
        )
    ]


def _parse_scores_array(
    raw_scores: Any,
    expected_count: int,
) -> list[float] | None:
    if not isinstance(raw_scores, list):
        return None

    if len(raw_scores) < expected_count:
        return None

    scores: list[float] = []

    for raw_score in raw_scores[:expected_count]:
        if isinstance(raw_score, bool):
            return None

        if not isinstance(raw_score, (int, float)):
            return None

        score = float(raw_score)

        if score < 0.0:
            score = 0.0

        if score > 1.0:
            score = 1.0

        scores.append(score)

    return scores


def parse_state_ratings(
    response: str,
    expected_count: int,
) -> list[str]:
    """
    解析 Qwen 对一批中间状态返回的评价。

    期望格式：

        {
          "ratings": [
            {
              "id": 1,
              "rating": "likely"
            },
            {
              "id": 2,
              "rating": "impossible"
            }
          ]
        }

    返回值顺序与状态 ID 顺序一致，例如：

        ["likely", "impossible"]

    容错策略：

        1. JSON 整体解析失败：
           所有状态返回 "maybe"。

        2. 缺少某个 ID：
           对应状态返回 "maybe"。

        3. rating 不在允许集合中：
           对应状态返回 "maybe"。

        4. 模型错误输出 Markdown JSON code block：
           自动删除 ```json 和 ```。
    """
    if expected_count < 0:
        raise ValueError(
            "expected_count must be non-negative"
        )

    if expected_count == 0:
        return []

    cleaned = _remove_code_fence(
        response.strip()
    )

    data = _parse_json_object(cleaned)

    if data is None:
        return _default_ratings(
            expected_count
        )

    raw_ratings = data.get("ratings")

    if not isinstance(raw_ratings, list):
        return _default_ratings(
            expected_count
        )

    parsed_by_id: dict[int, str] = {}

    for item in raw_ratings:
        if not isinstance(item, dict):
            continue

        state_id = item.get("id")
        rating = item.get("rating")

        # bool 是 int 的子类，因此要显式排除。
        if (
            not isinstance(state_id, int)
            or isinstance(state_id, bool)
        ):
            continue

        if not isinstance(rating, str):
            continue

        normalized_rating = (
            rating
            .strip()
            .lower()
        )

        if (
            normalized_rating
            not in VALID_STATE_RATINGS
        ):
            continue

        # 只接收当前批次范围内的 ID。
        if not (
            1
            <= state_id
            <= expected_count
        ):
            continue

        # 如果同一个 ID 重复出现，
        # 后出现的结果覆盖前面的结果。
        parsed_by_id[state_id] = (
            normalized_rating
        )

    return [
        parsed_by_id.get(
            state_id,
            "maybe",
        )
        for state_id in range(
            1,
            expected_count + 1,
        )
    ]


def _remove_code_fence(
    text: str,
) -> str:
    """
    删除模型可能附加的 Markdown code block。

    例如：

        ```json
        {"ratings": [...]}
        ```

    会变成：

        {"ratings": [...]}
    """
    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```\s*$",
        "",
        text,
    )

    return text.strip()


def _parse_json_object(
    text: str,
) -> dict[str, Any] | None:
    """
    尝试解析 JSON 对象。

    首先直接 json.loads。

    如果模型在 JSON 前后添加少量文本，
    再尝试截取第一个 { 到最后一个 }。
    """
    try:
        data = json.loads(text)

    except json.JSONDecodeError:
        first_brace = text.find("{")
        last_brace = text.rfind("}")

        if (
            first_brace == -1
            or last_brace == -1
            or first_brace >= last_brace
        ):
            return None

        json_fragment = text[
            first_brace:
            last_brace + 1
        ]

        try:
            data = json.loads(
                json_fragment
            )
        except json.JSONDecodeError:
            return None

    if not isinstance(data, dict):
        return None

    return data


def _default_ratings(
    count: int,
) -> list[str]:
    """
    模型评价无法解析时使用中性评价。

    选择 maybe 而不是 impossible，
    避免一次格式错误直接剪掉可能正确的搜索路径。
    """
    return [
        "maybe"
        for _ in range(count)
    ]


def _default_scores(
    count: int,
) -> list[float]:
    return [
        RATING_TO_SCORE["maybe"]
        for _ in range(count)
    ]
