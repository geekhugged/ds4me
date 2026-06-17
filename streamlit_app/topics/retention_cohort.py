import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.title("5. Retention, churn и cohort-анализ")
st.caption("Теория и формулы — pages/data-product.html, раздел 5.")

st.markdown(
    r"""
N-day retention для когорты:

$$ \text{Retention}_N = \frac{\#\{\text{активные в день } N\}}{\#\{\text{пользователи когорты на день } 0\}} $$

При постоянном churn rate $c$ в каждый период (упрощённая геометрическая модель):

$$ \text{Retention}(t) = (1-c)^t $$
"""
)

st.header("Модель retention curve: геометрическая модель против плато")

col1, col2 = st.columns(2)
with col1:
    churn_rate = st.slider("Постоянный churn rate за период c", 0.01, 0.5, 0.15, 0.01)
    n_periods = st.slider("Число периодов", 5, 52, 20, 1)
with col2:
    plateau = st.slider("Уровень плато (асимптота) realistic-модели", 0.0, 0.5, 0.15, 0.01)
    decay_speed = st.slider("Скорость затухания к плато", 0.05, 1.0, 0.3, 0.05)

t = np.arange(0, n_periods + 1)
geometric_retention = (1 - churn_rate) ** t
realistic_retention = plateau + (1 - plateau) * np.exp(-decay_speed * t)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(t, geometric_retention, color="indianred", label="геометрическая модель (постоянный churn)")
ax.plot(t, realistic_retention, color="seagreen", label="реалистичная модель (выход на плато)")
ax.axhline(plateau, color="gray", linestyle=":", label=f"плато = {plateau:.0%}")
ax.set_xlabel("период с момента старта когорты")
ax.set_ylabel("доля активных пользователей")
ax.set_ylim(0, 1.02)
ax.legend()
ax.set_title("Сравнение моделей retention curve")
st.pyplot(fig)

st.markdown(
    "Геометрическая модель с постоянным churn rate предполагает, что retention асимптотически "
    "стремится к нулю — на практике это редко так: пользователи, «выжившие» первые несколько "
    "периодов, обычно демонстрируют намного более низкий churn rate, и кривая выходит на плато "
    "выше нуля («smile point»). Высота этого плато — часто лучший индикатор product-market fit, "
    "чем абсолютные значения retention на первых днях."
)

st.header("Когортная таблица (heatmap)")

st.markdown(
    "Симулируем несколько когорт с разным качеством retention (например, из-за изменений в "
    "продукте или сезонности притока пользователей) и строим типичную когортную heatmap."
)

n_cohorts = st.slider("Число когорт (недель)", 4, 12, 8, 1)
n_weeks_tracked = st.slider("Число недель отслеживания", 4, 12, 8, 1)
base_plateau = st.slider("Базовый уровень плато retention", 0.05, 0.4, 0.15, 0.01, key="base_plateau")
trend = st.slider("Тренд изменения плато по когортам (п.п. за когорту)", -0.03, 0.03, 0.01, 0.005, format="%.3f")

rng = np.random.default_rng(42)
cohort_labels = [f"Когорта {i+1}" for i in range(n_cohorts)]
data = {}
for i in range(n_cohorts):
    cohort_plateau = max(0.01, base_plateau + trend * i + rng.normal(0, 0.01))
    decay = 0.4
    week_t = np.arange(0, n_weeks_tracked)
    curve = cohort_plateau + (1 - cohort_plateau) * np.exp(-decay * week_t)
    curve = np.clip(curve + rng.normal(0, 0.02, size=len(curve)), 0, 1)
    curve[0] = 1.0
    data[cohort_labels[i]] = curve

df = pd.DataFrame(data, index=[f"W{i}" for i in range(n_weeks_tracked)]).T

fig2, ax2 = plt.subplots(figsize=(9, 5))
im = ax2.imshow(df.values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
ax2.set_xticks(range(n_weeks_tracked))
ax2.set_xticklabels(df.columns)
ax2.set_yticks(range(n_cohorts))
ax2.set_yticklabels(df.index)
for i in range(n_cohorts):
    for j in range(n_weeks_tracked):
        ax2.text(j, i, f"{df.values[i,j]:.0%}", ha="center", va="center", fontsize=7,
                  color="black" if df.values[i, j] < 0.5 else "white")
fig2.colorbar(im, ax=ax2, label="retention")
ax2.set_title("Когортная heatmap retention по неделям")
st.pyplot(fig2)

if trend > 0:
    st.success(
        "Тренд положительный: более новые когорты удерживаются лучше — вероятно, продуктовые "
        "изменения повышают долгосрочную ценность для новых пользователей."
    )
elif trend < 0:
    st.warning(
        "Тренд отрицательный: более новые когорты удерживаются хуже старых — сигнал о "
        "деградации продукта или ухудшении качества привлекаемого трафика, который легко "
        "пропустить, если смотреть только на агрегированный DAU/MAU."
    )
else:
    st.info("Тренд нейтральный: качество удержания новых когорт стабильно относительно старых.")

st.header("Churn rate и прогноз размера активной базы")

active_base_now = st.number_input("Текущий размер активной базы", 100, 10_000_000, 100_000, 1000)
monthly_churn = st.slider("Месячный churn rate", 0.01, 0.3, 0.05, 0.01, key="monthly_churn")
monthly_new_users = st.number_input("Новые пользователи в месяц", 0, 1_000_000, 8000, 500)
months_forecast = st.slider("Горизонт прогноза, месяцев", 3, 36, 12, 1)

base = [active_base_now]
for _ in range(months_forecast):
    next_base = base[-1] * (1 - monthly_churn) + monthly_new_users
    base.append(next_base)

fig3, ax3 = plt.subplots(figsize=(7, 4))
ax3.plot(range(months_forecast + 1), base, color="navy", marker="o", markersize=3)
ax3.set_xlabel("месяц")
ax3.set_ylabel("размер активной базы")
ax3.set_title("Прогноз активной базы при постоянном churn и постоянном притоке")
st.pyplot(fig3)

equilibrium = monthly_new_users / monthly_churn if monthly_churn > 0 else float("inf")
st.markdown(
    f"При неизменных churn rate и притоке новых пользователей база стремится к равновесному "
    f"размеру **{equilibrium:,.0f}** пользователей "
    f"(решение уравнения $N = N(1-c) + \\text{{new}}$, то есть $N = \\text{{new}}/c$)."
)
