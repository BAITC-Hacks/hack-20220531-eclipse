"""Run the real E5 model against the supplied catalog and save measured results."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from data_loader import DATA_PATH, load_contractors
from embedding_service import (
    MODEL_NAME,
    encode_contractors,
    encode_description_sentences,
    load_embedding_model,
    split_description,
)
from recommender import calculate_ranking, check_candidate, filter_city_category, recommend


def main() -> None:
    records = load_contractors()
    started = perf_counter()
    model = load_embedding_model()
    load_seconds = perf_counter() - started
    started = perf_counter()
    matrix = encode_contractors(model, records)
    sentences = encode_description_sentences(model, (
        sentence for profile in records for sentence in split_description(profile.get("description"))
    ))
    encode_seconds = perf_counter() - started
    assert matrix.shape == (66, 384)
    assert np.isfinite(matrix).all()
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-5)

    dense = dict(city="Алматы", category="Ведущий", date="2026-11-14", event_format="корпоратив", budget_kzt=1_500_000, duration_hours=5, language="русский")
    rare = {**dense, "city": "Астана", "category": "Флорист", "budget_kzt": 350_000, "duration_hours": None}
    scenarios = {
        "dense": dense,
        "rare": rare,
        "no_results": {**rare, "budget_kzt": 100_000},
        "busy_date": {**rare, "date": "2026-11-02"},
        "category_not_found": {**rare, "category": "Несуществующая категория"},
    }
    report = {
        "model": MODEL_NAME,
        "dataset": DATA_PATH.name,
        "profiles": len(records),
        "matrix_shape": list(matrix.shape),
        "missing_descriptions": sum(not p.get("description") for p in records),
        "cached_sentences": len(sentences),
        "model_load_seconds": load_seconds,
        "catalog_encode_seconds": encode_seconds,
        "scenarios": {},
    }
    for name, query in scenarios.items():
        results = []
        times = []
        for _ in range(3):
            started = perf_counter()
            results.append(recommend(records, query, model, matrix, sentences))
            times.append(perf_counter() - started)
        assert results[0] == results[1] == results[2], f"Non-deterministic result: {name}"
        assert max(times) < 10, f"Slow warm query: {name}"
        result = results[0]
        report["scenarios"][name] = {
            "request": query, "status": result["status"], "stats": result["stats"],
            "query_seconds": times, "same_result_three_times": True,
            "top": [{"id": item["contractor"]["id"], "name": item["contractor"]["anon_name"], "score": item["score"], "ranking_details": item["ranking_details"], "explanation": item["explanation"]} for item in result["recommendations"]],
        }
    assert report["scenarios"]["dense"]["stats"]["eligible"] == 4
    assert report["scenarios"]["rare"]["top"][0]["id"] == "HK-90002"
    assert report["scenarios"]["no_results"]["status"] == "no_eligible_candidates"
    assert report["scenarios"]["busy_date"]["status"] == "no_eligible_candidates"

    candidates = [p for p in filter_city_category(records, dense["city"], dense["category"]) if not check_candidate(p, dense)]
    positions = {id(p): i for i, p in enumerate(records)}
    eligible_matrix = matrix[[positions[id(p)] for p in candidates]]
    for context in ("", "тимбилдинги и интерактивная программа", "деловой форум, официальная церемония", "живая музыка и юмор"):
        query = {**dense, "request_context": context}
        ranked = calculate_ranking(candidates, query, model, eligible_matrix, sentences)
        without_semantics = sorted(ranked, key=lambda item: (-(item["ranking_details"]["budget_points"] + item["ranking_details"]["duration_points"]), str(item["contractor"]["id"])))
        actual_ids = [item["contractor"]["id"] for item in ranked]
        baseline_ids = [item["contractor"]["id"] for item in without_semantics]
        if actual_ids != baseline_ids:
            report["semantic_order_effect"] = {"request": query, "with_semantics": actual_ids, "budget_duration_only": baseline_ids}
            break
    assert "semantic_order_effect" in report, "No ranking change observed in the real dense-category demos"

    output = Path(__file__).parent / "outputs" / "validation_report.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved report: {output}")


if __name__ == "__main__":
    main()
