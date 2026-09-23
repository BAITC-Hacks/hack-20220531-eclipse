"""Streamlit interface for the Smart Contractor Matching demo."""

from __future__ import annotations

from datetime import date
from time import perf_counter

import streamlit as st

from app_resources import (
    get_contractors,
    get_contractor_embeddings,
    get_embedding_model,
    get_sentence_embeddings,
)
from data_loader import DATA_PATH, unique_values
from embedding_service import (
    MODEL_NAME,
    build_contractor_embedding_text,
    split_description,
)
from recommender import LANGUAGE_ANY, recommend


st.set_page_config(page_title="Умный подбор подрядчиков", page_icon="🎉", layout="wide")


st.title("Умный подбор подрядчиков")
st.caption("Подбор по условиям события и описаниям опыта подрядчиков.")
debug = st.sidebar.checkbox("Показать детали ranking", value=False)

try:
    contractors = get_contractors(str(DATA_PATH), DATA_PATH.stat().st_mtime_ns)
except (OSError, ValueError) as exc:
    st.error(f"Не удалось загрузить каталог: {exc}")
    st.stop()
cities = unique_values(contractors, "city")

if not contractors:
    st.error("Каталог пуст. Проверьте файл данных в папке data.")
    st.stop()

try:
    embedding_model = get_embedding_model(MODEL_NAME)
    profile_texts = tuple(build_contractor_embedding_text(item) for item in contractors)
    contractor_embeddings = get_contractor_embeddings(profile_texts)
    sentences = tuple(dict.fromkeys(
        sentence for item in contractors for sentence in split_description(item.get("description"))
    ))
    sentence_embeddings = get_sentence_embeddings(sentences)
except Exception as exc:
    st.error(str(exc))
    st.info("При первом запуске нужны установленные зависимости и интернет для загрузки модели. Затем модель работает локально.")
    st.stop()

city_default = cities.index("Алматы") if "Алматы" in cities else 0
# Dependent selectors must rerun immediately; forms update only on submit.
city_column, category_column = st.columns(2)
with city_column:
    city = st.selectbox("Город", cities, index=city_default)
city_profiles = [item for item in contractors if item.get("city", "").casefold() == city.casefold()]
categories = unique_values(city_profiles, "categories")
if not categories:
    st.warning("В этом городе пока нет категорий подрядчиков.")
    st.stop()
with category_column:
    category_default = categories.index("Ведущий") if "Ведущий" in categories else 0
    category = st.selectbox("Категория подрядчика", categories, index=category_default)
category_profiles = [
    item for item in city_profiles
    if any(value.casefold() == category.casefold() for value in item.get("categories", []))
]
with st.form("matching_request"):
    left, right = st.columns(2)
    with left:
        event_formats = unique_values(category_profiles, "event_formats") or unique_values(contractors, "event_formats")
        format_default = event_formats.index("корпоратив") if "корпоратив" in event_formats else 0
        event_format = st.selectbox("Тип мероприятия", event_formats, index=format_default)
        event_date = st.date_input("Дата мероприятия", value=date(2026, 11, 14))
        request_context = st.text_area(
            "Что важно в подрядчике? (необязательно)",
            max_chars=300,
            placeholder="Например: интерактивная программа для большой команды",
        )
    with right:
        budget = st.number_input("Бюджет, ₸", min_value=0, value=1_500_000, step=50_000)
        duration_options: list[int | None] = [None, *range(1, 25)]
        duration = st.selectbox(
            "Длительность",
            duration_options,
            format_func=lambda value: "Не указывать" if value is None else f"{value} ч",
            index=5,
        )
        languages = unique_values(category_profiles, "languages")
        language_options = [LANGUAGE_ANY, *languages]
        language_default = language_options.index("русский") if "русский" in language_options else 0
        language = st.selectbox("Язык", language_options, index=language_default)

    submitted = st.form_submit_button("Подобрать", type="primary", use_container_width=True)


def _format_money(value: int | float | None) -> str:
    if value is None:
        return "не указана"
    return f"{int(value):,}".replace(",", " ")


def _show_summary(result: dict, city: str, category: str, event_date: date) -> None:
    stats = result["stats"]
    if result["status"] == "category_not_found":
        st.warning(f"В городе «{city}» в каталоге нет подрядчиков категории «{category}».")
        return

    total = stats["city_category_total"]
    st.markdown(f"**Найдено в категории:** {total} — {category} в городе {city}.")
    st.markdown(f"**На {event_date.strftime('%d.%m.%Y')}:**")
    counts = stats["rejection_counts"]
    labels = {
        "busy": "заняты на выбранную дату",
        "over_budget": "со стартовой ценой выше бюджета",
        "wrong_format": "не работают с выбранным форматом",
        "wrong_language": "не поддерживают выбранный язык",
        "duration_too_long": "не подходят по длительности",
    }
    active_reasons = [f"{count} {labels[key]}" for key, count in counts.items() if count]
    st.write("; ".join(active_reasons) + "." if active_reasons else "отказов по заданным условиям нет.")
    eligible = stats["eligible"]
    if result["status"] == "no_eligible_candidates":
        st.error(
            f"Подрядчики категории есть, но никто не проходит условия. "
            f"Профилей: {total}; по всем условиям проходят: 0. Причины отказа могут пересекаться."
        )
    else:
        shown = len(result["recommendations"])
        st.success(f"Под условия проходят {eligible} кандидатов. Подобрано {shown} подрядчиков.")


if submitted:
    request = {
        "city": city,
        "category": category,
        "date": event_date.isoformat(),
        "event_format": event_format,
        "budget_kzt": budget,
        "duration_hours": duration,
        "language": language,
        "request_context": request_context,
    }
    start = perf_counter()
    result = recommend(contractors, request, embedding_model, contractor_embeddings, sentence_embeddings)
    st.session_state["last_result"] = (request, result, perf_counter() - start)

saved_result = st.session_state.get("last_result")
if saved_result and (saved_result[0]["city"], saved_result[0]["category"]) != (city, category):
    st.session_state.pop("last_result", None)
    saved_result = None

if saved_result:
    saved_request, result, elapsed = saved_result
    st.caption("Результаты по последнему отправленному запросу.")
    _show_summary(result, city, category, date.fromisoformat(saved_request["date"]))
    if debug:
        with st.expander("Запрос и этапы подбора", expanded=True):
            st.write(f"Модель: {MODEL_NAME}; профили: {contractor_embeddings.shape}; время запроса: {elapsed:.3f} с.")
            st.json({"request": saved_request, "status": result["status"], "stats": result["stats"]})

    if result["status"] == "success":
        st.subheader("Рекомендации")
        columns = st.columns(len(result["recommendations"]))
        for column, item in zip(columns, result["recommendations"]):
            contractor = item["contractor"]
            with column:
                with st.container(border=True):
                    st.subheader(contractor.get("anon_name") or "Имя не указано")
                    st.caption(" · ".join(contractor.get("categories", [])))
                    st.write(f"**Город:** {contractor.get('city') or 'не указан'}")
                    st.write(f"**Стартовая цена:** {_format_money(contractor.get('price_from_kzt'))} ₸")
                    st.write(item["explanation"])
                    if debug:
                        details = item["ranking_details"]
                        st.table({
                            "Показатель": ["Semantic similarity", "Semantic points / 70", "Budget points / 20", "Duration points / 10", "Final score / 100"],
                            "Значение": [
                                f"{details['semantic_similarity']:.4f}",
                                f"{details['semantic_points']:.2f}",
                                f"{details['budget_points']:.2f}",
                                f"{details['duration_points']:.2f}",
                                f"{item['score']:.2f}",
                            ],
                        })
                        st.text(details["query_text"])
                        st.write("Фрагмент описания:", details["description_evidence"] or "Описание отсутствует.")
                    if contractor.get("price_imputed"):
                        st.caption("Цена восстановлена при подготовке датасета.")
                    if contractor.get("city_imputed"):
                        st.caption("Город восстановлен при подготовке датасета.")
                    if contractor.get("synthetic"):
                        st.warning("Synthetic profile", icon="🧪")
                    else:
                        st.caption("Dataset profile")
    elif result["status"] == "no_eligible_candidates":
        st.caption("Число отказов по разным причинам может пересекаться: один профиль мог не пройти несколько фильтров.")
