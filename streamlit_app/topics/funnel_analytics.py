import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.title("2. Продуктовая аналитика: события, воронки, сегментация")
st.caption("Теория и формулы — pages/data-product.html, раздел 2.")

st.markdown(
    r"""
Конверсия на шаге воронки между событиями $A$ и $B$:

$$ \text{CR}_{A \to B} = \frac{\#\{\text{пользователей, дошедших до } B\}}{\#\{\text{пользователей, дошедших до } A\}} $$

Сквозная конверсия воронки из $n$ шагов:

$$ \text{CR}_{\text{total}} = \prod_{i=1}^{n-1} \text{CR}_{i \to i+1} $$
"""
)

st.header("Конструктор воронки")

n_users = st.slider("Число пользователей на входе воронки", 1000, 200_000, 50_000, 1000)

default_steps = ["Открыл приложение", "Посмотрел товар", "Добавил в корзину", "Оформил заказ", "Оплатил", "Повторная покупка"]
n_steps = st.slider("Число шагов воронки", 3, 6, 5, 1)
steps = default_steps[:n_steps]

st.markdown("Задайте конверсию между соседними шагами:")
crs = []
cols = st.columns(n_steps - 1)
for i, col in enumerate(cols):
    with col:
        cr = st.slider(f"{steps[i]} → {steps[i+1]}", 0.05, 1.0, 0.5, 0.01, key=f"cr_{i}")
        crs.append(cr)

counts = [n_users]
for cr in crs:
    counts.append(counts[-1] * cr)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(steps, counts, color="teal")
for i, c in enumerate(counts):
    ax.text(i, c, f"{c:,.0f}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("число пользователей")
ax.set_title("Воронка продукта")
st.pyplot(fig)

total_cr = counts[-1] / counts[0]
st.metric("Сквозная конверсия воронки", f"{total_cr:.2%}")

drop_offs = [(steps[i], steps[i + 1], 1 - counts[i + 1] / counts[i]) for i in range(len(steps) - 1)]
worst = max(drop_offs, key=lambda x: x[2])
st.markdown(
    f"Самый «дырявый» шаг: **{worst[0]} → {worst[1]}** — теряется **{worst[2]:.1%}** "
    "пользователей. Улучшение этого шага даёт наибольший рычаг для роста сквозной конверсии."
)

st.header("Чувствительность сквозной конверсии к изменению одного шага")

step_to_improve = st.selectbox("Какой переход улучшить?", [f"{steps[i]} → {steps[i+1]}" for i in range(len(steps) - 1)])
idx = [f"{steps[i]} → {steps[i+1]}" for i in range(len(steps) - 1)].index(step_to_improve)

improve_range = np.linspace(0, 0.3, 30)
totals = []
for delta in improve_range:
    new_crs = crs.copy()
    new_crs[idx] = min(1.0, new_crs[idx] + delta)
    c = n_users
    for cr in new_crs:
        c *= cr
    totals.append(c / n_users)

fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.plot(improve_range, totals, color="darkorange")
ax2.set_xlabel(f"абсолютное увеличение конверсии на шаге «{step_to_improve}»")
ax2.set_ylabel("сквозная конверсия воронки")
ax2.set_title("Влияние улучшения одного шага на итоговую конверсию")
st.pyplot(fig2)

st.header("Сегментация: парадокс Симпсона на воронке")

st.markdown(
    "Демонстрация того, как общий тренд конверсии может скрывать противоположный тренд внутри "
    "сегментов, если меняется состав трафика между периодами (платный/органический канал)."
)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Период 1")
    paid_share_1 = st.slider("Доля платного трафика, период 1", 0.0, 1.0, 0.7, 0.05)
    cr_paid_1 = st.slider("Конверсия платного канала, период 1", 0.01, 0.5, 0.05, 0.01)
    cr_organic_1 = st.slider("Конверсия органического канала, период 1", 0.01, 0.5, 0.20, 0.01)
with col2:
    st.subheader("Период 2")
    paid_share_2 = st.slider("Доля платного трафика, период 2", 0.0, 1.0, 0.3, 0.05)
    cr_paid_2 = st.slider("Конверсия платного канала, период 2", 0.01, 0.5, 0.04, 0.01)
    cr_organic_2 = st.slider("Конверсия органического канала, период 2", 0.01, 0.5, 0.18, 0.01)

overall_1 = paid_share_1 * cr_paid_1 + (1 - paid_share_1) * cr_organic_1
overall_2 = paid_share_2 * cr_paid_2 + (1 - paid_share_2) * cr_organic_2

df = pd.DataFrame(
    {
        "Сегмент": ["Платный канал", "Органический канал", "Общая конверсия"],
        "Период 1": [cr_paid_1, cr_organic_1, overall_1],
        "Период 2": [cr_paid_2, cr_organic_2, overall_2],
    }
)
df["Изменение"] = df["Период 2"] - df["Период 1"]
st.dataframe(df.style.format({"Период 1": "{:.2%}", "Период 2": "{:.2%}", "Изменение": "{:+.2%}"}))

if (cr_paid_2 < cr_paid_1) and (cr_organic_2 < cr_organic_1) and (overall_2 > overall_1):
    st.warning(
        "Парадокс Симпсона: конверсия упала в обоих сегментах по отдельности, но общая "
        "конверсия выросла — потому что выросла доля более конвертящего органического канала. "
        "Без сегментации можно было бы решить, что продукт стал лучше конвертировать, хотя на "
        "самом деле качество трафика внутри каждого канала ухудшилось."
    )
elif (cr_paid_2 > cr_paid_1) and (cr_organic_2 > cr_organic_1) and (overall_2 < overall_1):
    st.warning(
        "Обратный парадокс Симпсона: конверсия выросла в обоих сегментах, но общая конверсия "
        "упала из-за изменения структуры трафика между периодами."
    )
else:
    st.info("Попробуйте подобрать значения так, чтобы тренд в сегментах был противоположен общему тренду.")
