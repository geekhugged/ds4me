import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

st.title("1. Проверка статистических гипотез: H0/H1, p-value, ошибки I и II рода")
st.caption("Теория и формулы — pages/experiment-design.html, раздел 1.")

st.markdown(
    r"""
При проверке гипотезы $H_0$ (нет эффекта) против $H_1$ (есть эффект) тестовая статистика
имеет одно распределение, если верна $H_0$, и другое (сдвинутое), если верна $H_1$.

- **Ошибка I рода ($\alpha$)** — площадь под кривой $H_0$ за порогом значимости (ложная тревога).
- **Ошибка II рода ($\beta$)** — площадь под кривой $H_1$ до порога значимости (пропущенный эффект).
- **Мощность ($1-\beta$)** — площадь под кривой $H_1$ за порогом.

Двигайте параметры ниже, чтобы увидеть, как $\alpha$, размер эффекта и размер выборки
влияют на эти площади.
"""
)

col1, col2, col3 = st.columns(3)
with col1:
    effect_size = st.slider("Истинный размер эффекта (стандартизованный, Cohen's d)", 0.0, 2.0, 0.5, 0.05)
with col2:
    alpha = st.slider("Уровень значимости α", 0.001, 0.2, 0.05, 0.001)
with col3:
    n = st.slider("Размер выборки (на группу)", 5, 1000, 50, 5)

# Стандартная ошибка разницы средних при дисперсии 1 в каждой группе (Cohen's d уже стандартизован)
se = np.sqrt(2 / n)

# Распределение статистики z под H0 и H1
mu0 = 0.0
mu1 = effect_size / se  # сдвиг в единицах SE для нестандартизованной z-статистики

z_crit = stats.norm.ppf(1 - alpha / 2)

x = np.linspace(-6, mu1 + 6, 1000)
pdf_h0 = stats.norm.pdf(x, loc=mu0, scale=1)
pdf_h1 = stats.norm.pdf(x, loc=mu1, scale=1)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x, pdf_h0, color="steelblue", label=r"распределение статистики при $H_0$")
ax.plot(x, pdf_h1, color="darkorange", label=r"распределение статистики при $H_1$")

# Ошибка I рода: площадь под H0 справа от z_crit (и слева от -z_crit)
x_alpha_right = x[x >= z_crit]
ax.fill_between(x_alpha_right, stats.norm.pdf(x_alpha_right, loc=mu0, scale=1),
                 color="steelblue", alpha=0.4, label=r"$\alpha$ (ошибка I рода)")
x_alpha_left = x[x <= -z_crit]
ax.fill_between(x_alpha_left, stats.norm.pdf(x_alpha_left, loc=mu0, scale=1),
                 color="steelblue", alpha=0.4)

# Ошибка II рода: площадь под H1 слева от z_crit
x_beta = x[x <= z_crit]
ax.fill_between(x_beta, stats.norm.pdf(x_beta, loc=mu1, scale=1),
                 color="darkorange", alpha=0.4, label=r"$\beta$ (ошибка II рода)")

# Мощность: площадь под H1 справа от z_crit
x_power = x[x >= z_crit]
ax.fill_between(x_power, stats.norm.pdf(x_power, loc=mu1, scale=1),
                 color="seagreen", alpha=0.4, label=r"мощность $1-\beta$")

ax.axvline(z_crit, color="red", linestyle="--", linewidth=1.5, label=f"порог значимости (z = {z_crit:.2f})")
ax.axvline(-z_crit, color="red", linestyle="--", linewidth=1.5)

ax.set_xlabel("значение тестовой статистики (z)")
ax.set_ylabel("плотность")
ax.legend(loc="upper right", fontsize=9)
ax.set_title("Распределения статистики под H0 и H1")
st.pyplot(fig)

# Вычисление мощности
power = 1 - (stats.norm.cdf(z_crit, loc=mu1, scale=1) - stats.norm.cdf(-z_crit, loc=mu1, scale=1))
beta = 1 - power

st.markdown(
    f"""
**При текущих параметрах:**

- $\\alpha$ = **{alpha:.3f}**
- $\\beta$ = **{beta:.3f}**
- мощность $1-\\beta$ = **{power:.3f}**
"""
)

st.header("Симуляция: частота ложных и истинных открытий")

st.markdown(
    "Запустим много повторов эксперимента при заданных параметрах и посчитаем, как часто "
    "мы отвергаем $H_0$ — отдельно для случая, когда $H_0$ верна (истинный эффект = 0), "
    "и для случая, когда верна $H_1$ (истинный эффект задан выше)."
)

n_sims = st.slider("Число симуляций", 100, 5000, 1000, 100)

if st.button("Запустить симуляцию"):
    rng = np.random.default_rng(0)

    # Под H0: истинный эффект = 0
    rejections_h0 = 0
    # Под H1: истинный эффект = effect_size
    rejections_h1 = 0

    for _ in range(n_sims):
        x1 = rng.normal(0, 1, size=n)
        x2_h0 = rng.normal(0, 1, size=n)
        x2_h1 = rng.normal(effect_size, 1, size=n)

        _, p_h0 = stats.ttest_ind(x1, x2_h0)
        _, p_h1 = stats.ttest_ind(x1, x2_h1)

        if p_h0 < alpha:
            rejections_h0 += 1
        if p_h1 < alpha:
            rejections_h1 += 1

    fpr = rejections_h0 / n_sims
    tpr = rejections_h1 / n_sims

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Доля отверженных H0, когда H0 верна (≈ α)", f"{fpr:.3f}")
    with col2:
        st.metric("Доля отверженных H0, когда верна H1 (≈ мощность)", f"{tpr:.3f}")

    st.markdown(
        f"""
Теоретическое значение $\\alpha$ = {alpha:.3f}, наблюдаемая частота ложных отвержений ≈ {fpr:.3f}.

Теоретическая мощность ≈ {power:.3f}, наблюдаемая частота истинных отвержений ≈ {tpr:.3f}.

Чем больше `n_sims`, тем точнее эмпирические частоты сходятся к теоретическим значениям.
"""
    )
