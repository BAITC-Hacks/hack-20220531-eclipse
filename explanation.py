"""Local, deterministic evidence extraction and explanation wording."""

from __future__ import annotations

import re
from typing import Any

from embedding_service import split_description


_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_QUERY_STOP_WORDS = {
    "query",
    "категория",
    "подрядчика",
    "формат",
    "мероприятия",
    "контекст",
    "event",
    "format",
}


def _values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = str(value).split("|")
    return [" ".join(str(item).split()) for item in items if str(item).strip()]


def find_relevant_description_part(description: str | None, query_text: str) -> str | None:
    """Pick a concise profile sentence using deterministic token overlap."""
    text = str(description or "").strip()
    if not text or not query_text.strip():
        return None
    sentences = [
        re.sub(r"\s+", " ", part).strip(" •\t\r\n")
        for part in re.split(r"(?<=[.!?])\s+|[\r\n]+|•", text)
    ]
    sentences = [sentence for sentence in sentences if len(sentence) >= 24]
    if not sentences:
        return None

    query_tokens = {
        token.replace("ё", "е")
        for token in _WORD_RE.findall(query_text.casefold())
        if token not in _QUERY_STOP_WORDS and len(token) >= 3
    }

    def overlap_score(sentence: str) -> int:
        sentence_tokens = {
            token.replace("ё", "е") for token in _WORD_RE.findall(sentence.casefold())
        }
        score = 0
        for query_token in query_tokens:
            if any(
                token == query_token
                or (
                    min(len(token), len(query_token)) >= 5
                    and token[:5] == query_token[:5]
                )
                for token in sentence_tokens
            ):
                score += 1
        return score

    best_index = max(range(len(sentences)), key=lambda index: (overlap_score(sentences[index]), -index))
    sentence = sentences[best_index]
    if len(sentence) > 155:
        sentence = sentence[:152].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"
    return sentence


def _format_kzt(value: Any) -> str:
    try:
        return f"{int(float(value)):,}".replace(",", " ")
    except (TypeError, ValueError):
        return "не указана"


def _money_fact(contractor: dict[str, Any], request: dict[str, Any]) -> str | None:
    price = contractor.get("price_from_kzt")
    budget = request.get("budget_kzt", request.get("budget"))
    if price is None:
        return None
    start_price = _format_kzt(price)
    try:
        remaining = max(0, int(float(budget) - float(price)))
    except (TypeError, ValueError):
        return f"Стартовая цена в профиле — {start_price} ₸; окончательную стоимость нужно уточнить у подрядчика."
    return (
        f"Цена начинается от {start_price} ₸, оставляя до бюджета около {_format_kzt(remaining)} ₸; "
        "итоговую стоимость нужно уточнить у подрядчика."
    )


def _candidate_facts(
    contractor: dict[str, Any], request: dict[str, Any], ranking_details: dict[str, Any]
) -> list[tuple[str, str]]:
    facts: list[tuple[str, str]] = []
    # Only quote an actual sentence, never infer skills from a similarity score.
    snippet = ranking_details.get("description_evidence")
    sentences = split_description(contractor.get("description"))
    if snippet not in sentences:
        snippet = None
    if "description_evidence" not in ranking_details:
        snippet = find_relevant_description_part(
            contractor.get("description"), str(ranking_details.get("query_text", ""))
        )
    if snippet:
        if len(snippet) > 220:
            snippet = snippet[:217].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"
        facts.append((f"description:{snippet.casefold()}", f"В описании профиля отмечено: «{snippet}»."))

    languages = _values(contractor.get("languages"))
    if languages:
        language_text = ", ".join(languages)
        facts.append((f"languages:{'|'.join(sorted(x.casefold() for x in languages))}", f"Поддерживаемые языки: {language_text}."))

    duration = request.get("duration_hours")
    if duration is not None:
        max_hours = contractor.get("max_hours")
        if max_hours is None:
            facts.append(("duration:independent", "В профиле не задан лимит часов на площадке; по правилам каталога такая услуга не ограничивается длительностью события."))
        else:
            facts.append((f"duration:{max_hours}", f"Заявленный лимит — до {max_hours} ч при запросе на {duration} ч."))

    money = _money_fact(contractor, request)
    if money:
        facts.append((f"price:{contractor.get('price_from_kzt')}", money))

    event_format = request.get("event_format", request.get("format"))
    if event_format:
        facts.append((f"format:{str(event_format).casefold()}", f"В профиле указан формат «{event_format}»."))
    return facts


def generate_explanation(
    contractor: dict[str, Any],
    request: dict[str, Any],
    ranking_details: dict[str, Any],
    used_evidence: set[str] | None = None,
) -> str:
    """Choose two concrete facts, preferring evidence not already used."""
    facts = _candidate_facts(contractor, request, ranking_details)
    if not facts:
        return "Профиль прошёл все обязательные фильтры каталога."
    used = used_evidence if used_evidence is not None else set()

    selected: list[tuple[str, str]] = []
    for fact in facts:
        if fact[0] not in used:
            selected.append(fact)
            used.add(fact[0])
        if len(selected) == 2:
            break
    if len(selected) < 2:
        for fact in facts:
            if fact not in selected:
                selected.append(fact)
            if len(selected) == 2:
                break
    return " ".join(sentence for _, sentence in selected)
