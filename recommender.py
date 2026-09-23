"""Deterministic filtering and ranking for contractor recommendations."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np

from data_loader import normalize_date
from embedding_service import (
    build_query_text,
    encode_description_sentences,
    encode_query,
    semantic_similarity,
    split_description,
)
from explanation import generate_explanation


LANGUAGE_ANY = "Не важно"
REJECTION_REASONS = (
    "busy",
    "over_budget",
    "wrong_format",
    "wrong_language",
    "duration_too_long",
)


def _normalized(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def _values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = str(value).split("|")
    return [" ".join(str(item).split()) for item in items if str(item).strip()]


def filter_city_category(
    contractors: list[dict[str, Any]], city: str, category: str
) -> list[dict[str, Any]]:
    """Look up a city/category pair before applying event constraints."""
    requested_city = _normalized(city)
    requested_category = _normalized(category)
    return [
        contractor
        for contractor in contractors
        if _normalized(contractor.get("city")) == requested_city
        and any(_normalized(item) == requested_category for item in _values(contractor.get("categories")))
    ]


def _request_date(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return normalize_date(value) or ""


def check_candidate(contractor: dict[str, Any], request: dict[str, Any]) -> list[str]:
    """Collect all hard-filter failures for a candidate; reasons can overlap."""
    reasons: list[str] = []
    requested_date = _request_date(request.get("date"))
    busy_dates = {_request_date(item) for item in _values(contractor.get("busy_dates"))}
    if requested_date and requested_date in busy_dates:
        reasons.append("busy")

    price = contractor.get("price_from_kzt")
    budget = request.get("budget_kzt", request.get("budget"))
    if price is not None and budget is not None:
        try:
            if float(price) > float(budget):
                reasons.append("over_budget")
        except (TypeError, ValueError):
            pass

    event_format = request.get("event_format", request.get("format"))
    if event_format and _normalized(event_format) not in {
        _normalized(item) for item in _values(contractor.get("event_formats"))
    }:
        reasons.append("wrong_format")

    language = request.get("language")
    if language and _normalized(language) != _normalized(LANGUAGE_ANY):
        if _normalized(language) not in {
            _normalized(item) for item in _values(contractor.get("languages"))
        }:
            reasons.append("wrong_language")

    duration = request.get("duration_hours")
    max_hours = contractor.get("max_hours")
    if duration is not None and max_hours is not None:
        try:
            if float(max_hours) < float(duration):
                reasons.append("duration_too_long")
        except (TypeError, ValueError):
            pass
    return reasons


def calculate_ranking(
    candidates: list[dict[str, Any]],
    request: dict[str, Any],
    embedding_model: Any,
    candidate_embeddings: np.ndarray,
    sentence_embeddings: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """Score already-eligible candidates and break ties by stable profile ID."""
    if not candidates:
        return []

    if len(candidate_embeddings) != len(candidates):
        raise ValueError("Для каждого допустимого профиля нужен ровно один embedding.")
    query_text = build_query_text(request)
    query_embedding = encode_query(embedding_model, request)
    similarities = semantic_similarity(query_embedding, candidate_embeddings)

    budget = request.get("budget_kzt", request.get("budget"))
    duration = request.get("duration_hours")
    scored: list[dict[str, Any]] = []
    for contractor, similarity in zip(candidates, similarities.tolist()):
        price = contractor.get("price_from_kzt")
        budget_score = 0.0
        if budget is not None and price is not None:
            try:
                budget_value = float(budget)
                if budget_value > 0:
                    budget_score = 20.0 * min(1.0, max(0.0, (budget_value - float(price)) / budget_value))
            except (TypeError, ValueError):
                pass

        duration_score = 0.0
        max_hours = contractor.get("max_hours")
        if duration is not None:
            if max_hours is None:
                # Null means the work does not depend on time at the venue.
                duration_score = 10.0
            else:
                try:
                    duration_score = min(10.0, max(0.0, 5.0 + float(max_hours) - float(duration)))
                except (TypeError, ValueError):
                    duration_score = 0.0

        description_relevance = min(1.0, max(0.0, float(similarity)))
        description_score = 70.0 * description_relevance
        score = description_score + budget_score + duration_score
        scored.append(
            {
                "contractor": contractor,
                "score": score,
                "ranking_details": {
                    "semantic_similarity": float(similarity),
                    "semantic_points": description_score,
                    "budget_points": budget_score,
                    "duration_points": duration_score,
                    # Keep the original keys for callers of the first version.
                    "description_similarity": float(similarity),
                    "description_relevance": description_relevance,
                    "description_score": description_score,
                    "budget_score": budget_score,
                    "duration_score": duration_score,
                    "query_text": query_text,
                },
            }
        )
    ranked = sorted(
        scored,
        key=lambda item: (-item["score"], str(item["contractor"].get("id", ""))),
    )
    # Explanations reuse the same query vector. In the app all sentences are cached;
    # standalone callers may omit the cache, encoding only Top 3 evidence in one batch.
    top_sentences = [
        split_description(item["contractor"].get("description")) for item in ranked[:3]
    ]
    evidence_vectors = dict(sentence_embeddings or {})
    missing = [s for sentences in top_sentences for s in sentences if s not in evidence_vectors]
    if missing:
        evidence_vectors.update(encode_description_sentences(embedding_model, missing))
    for item, sentences in zip(ranked[:3], top_sentences):
        details = item["ranking_details"]
        details["description_evidence"] = None
        details["evidence_similarity"] = None
        if sentences:
            scores = semantic_similarity(
                query_embedding, np.stack([evidence_vectors[s] for s in sentences])
            )
            best = int(np.argmax(scores))  # First sentence wins identical scores.
            details["description_evidence"] = sentences[best]
            details["evidence_similarity"] = float(scores[best])
    return ranked


def recommend(
    contractors: list[dict[str, Any]],
    request: dict[str, Any],
    embedding_model: Any | None = None,
    contractor_embeddings: np.ndarray | None = None,
    sentence_embeddings: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Return one of the explicit outcomes plus explanations and filter stats."""
    city = str(request.get("city", ""))
    category = str(request.get("category", ""))
    city_category = filter_city_category(contractors, city, category)
    stats: dict[str, Any] = {
        "city_category_total": len(city_category),
        "eligible": 0,
        "rejection_counts": {reason: 0 for reason in REJECTION_REASONS},
    }
    result: dict[str, Any] = {"stats": stats, "recommendations": [], "rejected": []}
    if not city_category:
        result["status"] = "category_not_found"
        return result

    eligible: list[dict[str, Any]] = []
    for contractor in city_category:
        reasons = check_candidate(contractor, request)
        if reasons:
            result["rejected"].append({"contractor": contractor, "reasons": reasons})
            for reason in reasons:
                stats["rejection_counts"][reason] += 1
        else:
            eligible.append(contractor)

    stats["eligible"] = len(eligible)
    if not eligible:
        result["status"] = "no_eligible_candidates"
        return result

    if embedding_model is None or contractor_embeddings is None:
        raise ValueError("Передайте модель embeddings и предварительно рассчитанные профили.")
    profile_matrix = np.asarray(contractor_embeddings, dtype=np.float32)
    if profile_matrix.ndim != 2 or profile_matrix.shape[0] != len(contractors):
        raise ValueError("Матрица embeddings должна соответствовать полному списку профилей.")
    profile_positions = {id(profile): index for index, profile in enumerate(contractors)}
    eligible_positions = [profile_positions[id(profile)] for profile in eligible]
    eligible_embeddings = profile_matrix[eligible_positions]

    ranked = calculate_ranking(
        eligible, request, embedding_model, eligible_embeddings, sentence_embeddings
    )
    recommendations: list[dict[str, Any]] = []
    used_evidence: set[str] = set()
    for item in ranked[:3]:
        contractor = item["contractor"]
        explanation = generate_explanation(
            contractor,
            request,
            item["ranking_details"],
            used_evidence=used_evidence,
        )
        recommendations.append(
            {
                "contractor": contractor,
                "score": item["score"],
                "ranking_details": item["ranking_details"],
                "explanation": explanation,
            }
        )
    result["status"] = "success"
    result["recommendations"] = recommendations
    return result
