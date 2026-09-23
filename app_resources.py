"""Streamlit caches, kept separate from the deterministic recommendation service."""

import streamlit as st

from data_loader import load_contractors
from embedding_service import (
    MODEL_NAME,
    encode_description_sentences,
    encode_texts,
    load_embedding_model,
)


@st.cache_data(show_spinner=False, max_entries=4)
def get_contractors(path: str, modified_ns: int) -> list[dict]:
    """The file timestamp invalidates the normalized catalog after an edit."""
    return load_contractors(path)


@st.cache_resource(show_spinner="Загрузка модели семантического поиска…")
def get_embedding_model(model_name: str):
    # A required argument prevents separate cache entries for () and (MODEL_NAME,).
    return load_embedding_model(model_name)


@st.cache_data(show_spinner="Подготавливаем каталог для поиска…", max_entries=4)
def get_contractor_embeddings(profile_texts: tuple[str, ...], model_name: str = MODEL_NAME):
    """Both model name and ordered semantic texts participate in the cache key."""
    return encode_texts(get_embedding_model(model_name), profile_texts)


@st.cache_data(show_spinner="Подготавливаем фрагменты описаний…", max_entries=4)
def get_sentence_embeddings(sentences: tuple[str, ...], model_name: str = MODEL_NAME):
    return encode_description_sentences(get_embedding_model(model_name), sentences)
