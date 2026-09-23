"""Pretrained multilingual embeddings for contractor and request text."""

from __future__ import annotations

import os
import re
from typing import Any, Iterable

import numpy as np


MODEL_NAME = "intfloat/multilingual-e5-small"


def load_embedding_model(model_name: str = MODEL_NAME) -> Any:
    """Prefer local weights; download only on a cache miss. Never train the model."""
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Не установлен sentence-transformers. Установите зависимости из requirements.txt."
        ) from exc
    # Small CPU batches can be slower when PyTorch starts too many worker threads.
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    try:
        try:
            model = SentenceTransformer(model_name, device="cpu", local_files_only=True)
        except OSError:
            model = SentenceTransformer(model_name, device="cpu")
        model.eval()
        return model
    except Exception as exc:
        raise RuntimeError(
            f"Не удалось загрузить модель {model_name}. При первом запуске нужен доступ "
            "к Hugging Face, чтобы скачать pretrained weights."
        ) from exc


def _field_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, set):
        values = sorted(value, key=str)
    elif isinstance(value, (list, tuple)):
        values: Iterable[Any] = value
    else:
        values = str(value).split("|")
    return [" ".join(str(item).split()) for item in values if str(item).strip()]


def build_query_text(request: dict[str, Any]) -> str:
    """Build a short E5 query from semantic intent, not hard-filter values."""
    category = " ".join(str(request.get("category") or "").split())
    event_format = " ".join(
        str(request.get("event_format", request.get("format")) or "").split()
    )
    parts: list[str] = []
    if category:
        parts.append(f"Категория подрядчика: {category}.")
    if event_format:
        parts.append(f"Формат мероприятия: {event_format}.")

    context = " ".join(
        str(request.get("request_context", request.get("context")) or "").split()
    )
    if context:
        parts.append(f"Контекст: {context[:300]}.")
    return "query: " + " ".join(parts)


def build_contractor_embedding_text(contractor: dict[str, Any]) -> str:
    """Include specialization and experience, excluding operational constraints."""
    categories = ", ".join(_field_values(contractor.get("categories"))) or "не указана"
    formats = ", ".join(_field_values(contractor.get("event_formats"))) or "не указаны"
    description = " ".join(str(contractor.get("description") or "").split())
    return (
        f"passage: Описание: {description}. "
        f"Категория: {categories}. Форматы мероприятий: {formats}."
    )


def encode_texts(model: Any, texts: Iterable[str]) -> np.ndarray:
    """Encode text with unit-normalized vectors suitable for dot-product cosine."""
    batch = list(texts)
    if not batch:
        return np.empty((0, 0), dtype=np.float32)
    vectors = model.encode(
        batch,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=32,
    )
    result = np.asarray(vectors, dtype=np.float32)
    if result.ndim == 1:
        result = result.reshape(1, -1)
    if result.ndim != 2 or len(result) != len(batch) or not np.isfinite(result).all():
        raise ValueError("Модель вернула некорректные embedding-векторы.")
    return result


def split_description(description: str | None) -> tuple[str, ...]:
    """Keep original sentences as evidence, including short RU/KZ/EN sentences."""
    parts = re.split(r"(?<=[.!?])\s+|[\r\n]+|•", str(description or "").strip())
    sentences = (" ".join(part.strip(" •\t\r\n").split()) for part in parts)
    return tuple(sentence for sentence in sentences if sentence)


def encode_description_sentences(
    model: Any, sentences: Iterable[str]
) -> dict[str, np.ndarray]:
    """Encode distinct evidence passages in one batch; cache this with the catalog."""
    unique = tuple(dict.fromkeys(sentences))
    vectors = encode_texts(model, (f"passage: {sentence}" for sentence in unique))
    return dict(zip(unique, vectors))


def encode_query(model: Any, request: dict[str, Any]) -> np.ndarray:
    """Encode one request using the multilingual E5 query prefix."""
    return encode_texts(model, [build_query_text(request)])[0]


def encode_contractors(
    model: Any, contractors: list[dict[str, Any]]
) -> np.ndarray:
    """Encode profile passages once; cache the returned matrix in Streamlit."""
    texts = [build_contractor_embedding_text(item) for item in contractors]
    return encode_texts(model, texts)


def semantic_similarity(
    query_embedding: np.ndarray, contractor_embeddings: np.ndarray
) -> np.ndarray:
    """Compute cosine similarity by dot product on normalized embeddings."""
    query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    passages = np.asarray(contractor_embeddings, dtype=np.float32)
    if passages.size == 0:
        return np.empty((0,), dtype=np.float32)
    if passages.ndim != 2 or passages.shape[1] != query.shape[0]:
        raise ValueError("Размерности embedding-векторов запроса и профилей не совпадают.")
    return passages @ query
