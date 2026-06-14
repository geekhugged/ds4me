import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

st.title("2. Доверительные интервалы и их интерпретация")
st.caption("Теория и формулы — pages/experiment-design.html, раздел 2.")

st.markdown(
    r"""
95%-й доверительный интервал для среднего строится как

$$ \text{CI} = \bar{X} \pm z_{1-\alpha/2} \cdot \frac{\sigma}{\sqrt{n}} $$

Корректная интерпретация: если повторить эксперимент много раз и каждый раз построить такой
интервал, то примерно 95% из этих интервалов накроют истинное значение параметра $\mu$.
Это свойство **процедуры**, а не утверждение о вероятности для одного конкретного интервала.

Симуляция ниже наглядно показывает это свойство — **coverage** (долю интервалов, накрывших
истинное значение).
"""
)

col1, col2, col3 = st.columns(3)
with col1:
    true_mu = st.slider("Истинное среднее популяции (μ)", -5.0, 5.0, 0.0, 0.5)
with col2:
    sigma = st.slider("Стандартное отклонение популяции (σ)", 0.5, 5.0, 1.0, 0.1)
with col3:
    n = st.slider("Размер выборки (n)", 5, 500, 30, 5)

confidence = st.select_slider("Уровень доверия", options=[0.80, 0.90, 0.95, 0.99], value=0.95)
alpha = 1 - confidence
z_crit = stats.norm.ppf(1 - alpha / 2)

n_intervals = st.slider("Число выборок (интервалов) для отображения", 10, 200, 50, 10)

rng = np.random.default_rng(42)

samples = rng.normal(loc=true_mu, scale=sigma, size=(n_intervals, n))
means = samples.mean(axis=1)
se = sigma / np.sqrt(n)
lower = means - z_crit * se
upper = means + z_crit * se

covers = (lower <= true_mu) & (upper >= true_mu)
coverage = covers.mean()

fig, ax = plt.subplots(figsize=(8, max(4, n_intervals * 0.08)))
colors = np.where(covers, "seagreen", "crimson")
y = np.arange(n_intervals)
ax.hlines(y, lower, upper, color=colors, alpha=0.7, linewidth=1.5)
ax.scatter(means, y, color=colors, s=8, zorder=3)
ax.axvline(true_mu, color="black", linestyle="--", linewidth=1.5, label=f"истинное μ = {true_mu}")
ax.set_yticks([])
ax.set_xlabel("значение")
ax.set_title(f"{int(confidence*100)}% доверительные интервалы для {n_intervals} выборок")
ax.legend(loc="upper right")
st.pyplot(fig)

st.markdown(
    f"""
**Наблюдаемый coverage:** {coverage:.1%} из {n_intervals} построенных интервалов накрыли
истинное значение μ = {true_mu}. Зелёные интервалы — накрыли, красные — не накрыли.

Теоретически при уровне доверия {int(confidence*100)}% ожидаем покрытие примерно
{int(confidence*100)}% (с разбросом из-за конечного числа интервалов).
"""
)

st.header("Большая симуляция: coverage при многих повторениях")

n_sims = st.slider("Число симуляций для оценки coverage", 100, 10000, 2000, 100)

if st.button("Запустить большую симуляцию"):
    rng2 = np.random.default_rng(1)
    big_samples = rng2.normal(loc=true_mu, scale=sigma, size=(n_sims, n))
    big_means = big_samples.mean(axis=1)
    big_lower = big_means - z_crit * se
    big_upper = big_means + z_crit * se
    big_covers = (big_lower <= true_mu) & (big_upper >= true_mu)
    big_coverage = big_covers.mean()

    st.metric(
        f"Эмпирический coverage при {n_sims} повторениях",
        f"{big_coverage:.4f}",
        delta=f"теор. {confidence:.2f}",
    )

    st.markdown(
        "При большом числе повторений эмпирический coverage сходится к заявленному уровню "
        "доверия — это и есть формальное определение доверительного интервала."
    )
