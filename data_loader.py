"""Load and normalize contractor records from CSV or JSON Lines."""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DATA_PATH = Path(__file__).parent / "data" / "hackathon-dataset-anonymized.csv"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _split_values(value: Any) -> list[str]:
    if _is_missing(value):
        return []
    if isinstance(value, (list, tuple, set)):
        parts: Iterable[Any] = value
    else:
        parts = str(value).split("|")
    result: list[str] = []
    seen: set[str] = set()
    for part in parts:
        text = " ".join(str(part).split())
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if _is_missing(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().casefold() in {"true", "1", "yes", "да"}


def _as_number(value: Any) -> int | float | None:
    if _is_missing(value) or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def normalize_date(value: Any) -> str | None:
    """Return a date as ISO YYYY-MM-DD, preserving unknown values defensively."""
    if _is_missing(value) or not str(value).strip():
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            return text


def normalize_contractor(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize one CSV/JSONL row to the application data contract."""
    record = dict(raw)
    for field in ("categories", "event_formats", "languages"):
        record[field] = _split_values(record.get(field))
    record["busy_dates"] = [
        normalized
        for item in _split_values(record.get("busy_dates"))
        if (normalized := normalize_date(item)) is not None
    ]
    for field in ("price_from_kzt", "max_hours"):
        record[field] = _as_number(record.get(field))
    for field in ("synthetic", "city_imputed", "price_imputed"):
        record[field] = _as_bool(record.get(field))
    for field in ("id", "anon_name", "city", "description"):
        value = record.get(field)
        record[field] = "" if _is_missing(value) else str(value).strip()
    return record


def load_contractors(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Read the provided CSV, or JSONL when a JSON Lines path is supplied."""
    source = Path(path) if path is not None else DATA_PATH
    if not source.exists():
        raise FileNotFoundError(f"Не найден датасет подрядчиков: {source}")

    suffix = source.suffix.casefold()
    if suffix == ".csv":
        frame = pd.read_csv(source, encoding="utf-8-sig", keep_default_na=True)
        rows = frame.to_dict(orient="records")
    elif suffix in {".jsonl", ".ndjson"}:
        rows = []
        with source.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Некорректный JSONL в строке {line_number}: {exc}") from exc
                    if isinstance(item, dict):
                        rows.append(item)
    else:
        raise ValueError("Поддерживаются файлы .csv, .jsonl и .ndjson")

    return [normalize_contractor(row) for row in rows]


def unique_values(contractors: list[dict[str, Any]], field: str) -> list[str]:
    """Return stable, case-insensitive unique scalar/list values."""
    values: dict[str, str] = {}
    for contractor in contractors:
        raw = contractor.get(field)
        items = raw if isinstance(raw, list) else [raw]
        for item in items:
            if item is None:
                continue
            text = " ".join(str(item).split())
            if text:
                values.setdefault(text.casefold(), text)
    return sorted(values.values(), key=lambda value: (value.casefold(), value))
