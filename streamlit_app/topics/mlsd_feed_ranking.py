import numpy as np
import pandas as pd
import streamlit as st

st.title("7.3 Ранжирование ленты соцсети (News Feed Ranking)")
st.caption(
    "Кейс — pages/ml-system-design-interview.html, раздел 3 "
    "(«Ранжирование ленты соцсети (News Feed Ranking)»)."
)

st.markdown(
    r"""
Многозадачная (multi-task) модель ленты предсказывает для каждого поста несколько вероятностей
взаимодействия: $P(\text{like})$, $P(\text{comment})$, $P(\text{share})$, $P(\text{click})$,
$P(\text{hide})$. Итоговый "value score" — взвешенная сумма:

$$ \text{score} = \sum_k w_k \cdot P(\text{action}_k) $$

где веса $w_k$ задаёт продуктовая команда: например, комментарий ценнее лайка
($w_{comment} > w_{like}$), а "hide"/"report" — сильный негативный сигнал ($w_{hide} \ll 0$).

Ниже — симуляция ленты из постов с предсказанными вероятностями взаимодействий. Меняйте веса
$w_k$ и смотрите, как меняется итоговый ранкинг — это иллюстрирует, почему выбор весов value
model — это продуктовое решение с прямым влиянием на то, какой контент пользователи видят
чаще.
"""
)

st.header("Симуляция ранжирования ленты")

rng = np.random.default_rng(7)
n_posts = st.slider("Число кандидатов в ленте", 10, 60, 25, 5)

content_types = ["Текст", "Фото", "Видео", "Ссылка", "Reels"]

df = pd.DataFrame(
    {
        "post_id": [f"post_{i}" for i in range(n_posts)],
        "author": rng.choice([f"Автор {i}" for i in range(1, 9)], n_posts),
        "type": rng.choice(content_types, n_posts),
        "age_min": rng.exponential(60, n_posts).round(1),
        "p_like": rng.beta(3, 10, n_posts).round(3),
        "p_comment": rng.beta(1, 30, n_posts).round(3),
        "p_share": rng.beta(1, 50, n_posts).round(3),
        "p_click": rng.beta(2, 6, n_posts).round(3),
        "p_hide": rng.beta(1, 60, n_posts).round(3),
    }
)

st.subheader("Веса value model")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    w_like = st.slider("w_like", 0.0, 5.0, 1.0, 0.1)
with col2:
    w_comment = st.slider("w_comment", 0.0, 10.0, 4.0, 0.1)
with col3:
    w_share = st.slider("w_share", 0.0, 10.0, 5.0, 0.1)
with col4:
    w_click = st.slider("w_click", 0.0, 5.0, 0.5, 0.1)
with col5:
    w_hide = st.slider("w_hide (штраф)", 0.0, 50.0, 20.0, 1.0)

st.subheader("Time decay")
decay_halflife = st.slider(
    "Период полураспада свежести поста (минут)", 5, 300, 60, 5,
    help="Скор умножается на 0.5^(age/halflife) — старые посты получают пониженный приоритет",
)

decay_factor = 0.5 ** (df["age_min"] / decay_halflife)

df["score"] = decay_factor * (
    w_like * df["p_like"]
    + w_comment * df["p_comment"]
    + w_share * df["p_share"]
    + w_click * df["p_click"]
    - w_hide * df["p_hide"]
)

st.subheader("Diversity-ограничение")
max_per_author = st.slider("Максимум постов одного автора подряд в ленте", 1, 5, 2)
top_n = st.slider("Размер финальной ленты (top-N)", 5, 25, 10)

ranked = df.sort_values("score", ascending=False).reset_index(drop=True)


def apply_diversity(ranked_df, max_per_author, top_n):
    selected = []
    author_counts = {}
    leftover = []
    for _, row in ranked_df.iterrows():
        a = row["author"]
        if author_counts.get(a, 0) < max_per_author and len(selected) < top_n:
            selected.append(row)
            author_counts[a] = author_counts.get(a, 0) + 1
        else:
            leftover.append(row)
    i = 0
    while len(selected) < top_n and i < len(leftover):
        selected.append(leftover[i])
        i += 1
    return pd.DataFrame(selected).reset_index(drop=True)


final_feed = apply_diversity(ranked, max_per_author, top_n)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Топ по чистому скору")
    st.dataframe(
        ranked.head(top_n)[["post_id", "author", "type", "age_min", "score"]].style.format(
            {"score": "{:.3f}", "age_min": "{:.1f}"}
        ),
        hide_index=True,
        use_container_width=True,
    )
with col2:
    st.subheader("После diversity re-ranking")
    st.dataframe(
        final_feed[["post_id", "author", "type", "age_min", "score"]].style.format(
            {"score": "{:.3f}", "age_min": "{:.1f}"}
        ),
        hide_index=True,
        use_container_width=True,
    )

# MSI (meaningful social interactions) - approximate proxy
msi_before = (
    ranked.head(top_n)["p_like"]
    + ranked.head(top_n)["p_comment"] * 4
    + ranked.head(top_n)["p_share"] * 5
).sum()
msi_after = (
    final_feed["p_like"] + final_feed["p_comment"] * 4 + final_feed["p_share"] * 5
).sum()
hide_before = ranked.head(top_n)["p_hide"].sum()
hide_after = final_feed["p_hide"].sum()
unique_authors_before = ranked.head(top_n)["author"].nunique()
unique_authors_after = final_feed["author"].nunique()

m1, m2, m3 = st.columns(3)
m1.metric("Ожидаемый MSI-проxy (без diversity)", f"{msi_before:.2f}", delta=f"{msi_after - msi_before:+.2f}")
m2.metric("Ожидаемый negative feedback (без diversity)", f"{hide_before:.3f}", delta=f"{hide_after - hide_before:+.3f}")
m3.metric("Уникальных авторов: было / стало", f"{unique_authors_before} / {unique_authors_after}")

st.markdown(
    """
**На что обратить внимание:**
- Рост `w_comment`/`w_share` относительно `w_like` смещает ленту в сторону постов, провоцирующих
  более "глубокое" вовлечение — это часто коррелирует с типом контента (например, видео/Reels
  чаще набирают share).
- Увеличение `w_hide` сильно понижает посты с даже небольшой вероятностью негативной реакции —
  на практике это инструмент борьбы с "borderline"/спам-контентом.
- Diversity-ограничение по автору может слегка снижать MSI-proxy, но повышает разнообразие
  источников в ленте (защита от echo chamber).
- `decay_halflife` иллюстрирует, насколько агрессивно лента отдаёт приоритет свежему контенту.
"""
)

st.header("Back-of-the-envelope: нагрузка")

col1, col2, col3 = st.columns(3)
with col1:
    dau_m = st.slider("DAU, млн пользователей", 10, 2000, 1000, 10)
with col2:
    opens_per_day = st.slider("Открытий ленты в день на пользователя", 1, 30, 10)
with col3:
    candidates_per_request = st.slider("Кандидатов на heavy ranking на запрос", 50, 1000, 300, 50)

total_requests = dau_m * 1e6 * opens_per_day
avg_qps = total_requests / 86400
peak_qps = avg_qps * 2.5

m1, m2, m3 = st.columns(3)
m1.metric("Запросов ленты в день", f"{total_requests:,.0f}")
m2.metric("Средний QPS", f"{avg_qps:,.0f}")
m3.metric("Пиковый QPS (×2.5)", f"{peak_qps:,.0f}")

st.metric(
    "Multi-task инференсов в секунду на пике",
    f"{peak_qps * candidates_per_request:,.0f}",
    help="QPS × число кандидатов на heavy ranking — нагрузка на MMoE-модель",
)
