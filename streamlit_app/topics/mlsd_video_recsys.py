import numpy as np
import pandas as pd
import streamlit as st

st.title("7.1 Видео-рекомендации для домашней страницы (YouTube-style)")
st.caption(
    "Кейс — pages/ml-system-design-interview.html, раздел 1 "
    "(«Видео-рекомендации для домашней страницы»)."
)

st.markdown(
    r"""
Двухстадийный pipeline рекомендаций: **candidate generation** (быстрый retrieval сотен
кандидатов из миллиардного каталога через ANN-поиск по embedding'ам) → **ranking**
(многозадачная модель оценивает $P(\text{click})$, $E[\text{watch\_time}]$, $P(\text{like})$,
$P(\text{dislike})$ для каждого кандидата) → **re-ranking** (бизнес-правила, diversity,
explore-бонус для нового контента).

Итоговый скор ранжирования — взвешенная сумма предсказаний:

$$ \text{score} = w_{click} \cdot P(\text{click}) + w_{watch} \cdot E[\text{watch\_time}]
   + w_{like} \cdot P(\text{like}) - w_{dislike} \cdot P(\text{dislike}) $$

Ниже — симуляция: для пула видео-кандидатов задаются предсказанные вероятности/значения по
каждой задаче, а вы управляете весами $w_k$ итоговой формулы и diversity-ограничением, чтобы
увидеть, как меняется финальный топ-N ленты.
"""
)

st.header("Симуляция: от скорингов к финальной ленте")

rng = np.random.default_rng(42)
n_candidates = st.slider("Число кандидатов после ranking-стадии", 10, 100, 30, 5)

categories = ["Музыка", "Игры", "Новости", "Юмор", "Образование", "Спорт"]
channels = [f"Канал {i}" for i in range(1, 11)]

df = pd.DataFrame(
    {
        "video_id": [f"v{i}" for i in range(n_candidates)],
        "channel": rng.choice(channels, n_candidates),
        "category": rng.choice(categories, n_candidates),
        "p_click": rng.beta(2, 8, n_candidates).round(3),
        "watch_time_min": (rng.gamma(2, 3, n_candidates)).round(2),
        "p_like": rng.beta(2, 15, n_candidates).round(3),
        "p_dislike": rng.beta(1, 40, n_candidates).round(3),
        "is_new": rng.random(n_candidates) < 0.15,
    }
)

st.subheader("Веса итогового скора")
col1, col2, col3, col4 = st.columns(4)
with col1:
    w_click = st.slider("w_click", 0.0, 5.0, 1.0, 0.1)
with col2:
    w_watch = st.slider("w_watch", 0.0, 5.0, 1.0, 0.1)
with col3:
    w_like = st.slider("w_like", 0.0, 5.0, 1.0, 0.1)
with col4:
    w_dislike = st.slider("w_dislike (штраф)", 0.0, 20.0, 5.0, 0.5)

explore_bonus = st.slider(
    "Explore-бонус для нового контента (is_new)", 0.0, 5.0, 0.5, 0.1,
    help="Аддитивный бонус к скору новых видео — компенсация холодного старта",
)

df["score"] = (
    w_click * df["p_click"]
    + w_watch * df["watch_time_min"] / df["watch_time_min"].max()
    + w_like * df["p_like"]
    - w_dislike * df["p_dislike"]
    + np.where(df["is_new"], explore_bonus, 0.0)
)

st.subheader("Diversity-ограничение")
max_per_channel = st.slider(
    "Максимум видео одного канала в финальной ленте подряд (re-ranking)", 1, 5, 2,
)
top_n = st.slider("Размер финальной ленты (top-N)", 5, 30, 15)

ranked = df.sort_values("score", ascending=False).reset_index(drop=True)


def apply_diversity(ranked_df, max_per_channel, top_n):
    selected = []
    channel_counts = {}
    leftover = []
    for _, row in ranked_df.iterrows():
        ch = row["channel"]
        if channel_counts.get(ch, 0) < max_per_channel and len(selected) < top_n:
            selected.append(row)
            channel_counts[ch] = channel_counts.get(ch, 0) + 1
        else:
            leftover.append(row)
    # докидываем оставшиеся места, если не хватило (игнорируя ограничение)
    i = 0
    while len(selected) < top_n and i < len(leftover):
        selected.append(leftover[i])
        i += 1
    return pd.DataFrame(selected).reset_index(drop=True)


final_feed = apply_diversity(ranked, max_per_channel, top_n)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Ranking без diversity (топ по скору)")
    st.dataframe(
        ranked.head(top_n)[
            ["video_id", "channel", "category", "score", "is_new"]
        ].style.format({"score": "{:.3f}"}),
        hide_index=True,
        use_container_width=True,
    )
with col2:
    st.subheader("Финальная лента (после re-ranking)")
    st.dataframe(
        final_feed[["video_id", "channel", "category", "score", "is_new"]].style.format(
            {"score": "{:.3f}"}
        ),
        hide_index=True,
        use_container_width=True,
    )

unique_channels_before = ranked.head(top_n)["channel"].nunique()
unique_channels_after = final_feed["channel"].nunique()
n_new_before = int(ranked.head(top_n)["is_new"].sum())
n_new_after = int(final_feed["is_new"].sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Уникальных каналов (без diversity)", unique_channels_before)
m2.metric("Уникальных каналов (с diversity)", unique_channels_after)
m3.metric("Новых видео (без explore)", n_new_before)
m4.metric("Новых видео (с explore)", n_new_after)

st.markdown(
    """
**На что обратить внимание:**
- Увеличение `w_dislike` сильно понижает кликбейтные/раздражающие видео, даже при высоком CTR.
- `explore_bonus` помогает новым видео (холодный старт) пробиться в выдачу для сбора сигналов.
- Diversity-ограничение по каналу повышает разнообразие финальной ленты, иногда жертвуя
  топ-кандидатами по чистому скору — это типичный trade-off relevance vs diversity.
"""
)

st.header("Back-of-the-envelope: нагрузка на serving")

col1, col2, col3 = st.columns(3)
with col1:
    dau = st.slider("Активных пользователей в день (DAU), млн", 10, 2000, 500, 10)
with col2:
    requests_per_user = st.slider("Запросов главной страницы на пользователя/день", 1, 20, 3)
with col3:
    n_cand_rank = st.slider("Кандидатов на ranking-стадии на запрос", 50, 1000, 300, 50)

total_requests_per_day = dau * 1e6 * requests_per_user
avg_qps = total_requests_per_day / 86400
peak_qps = avg_qps * 3  # типичный peak/avg multiplier

m1, m2, m3 = st.columns(3)
m1.metric("Запросов в день", f"{total_requests_per_day:,.0f}")
m2.metric("Средний QPS", f"{avg_qps:,.0f}")
m3.metric("Пиковый QPS (×3)", f"{peak_qps:,.0f}")

forward_passes_per_sec = peak_qps * n_cand_rank
st.metric(
    "Forward-pass ranking-модели в секунду на пике",
    f"{forward_passes_per_sec:,.0f}",
    help="QPS × число кандидатов на ranking-стадии — определяет требования к GPU/TPU serving-кластеру",
)
