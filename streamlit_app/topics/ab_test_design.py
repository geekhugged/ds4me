import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

st.title("3. Дизайн A/B-теста: рандомизация, MDE и расчёт размера выборки/мощности")
st.caption("Теория и формулы — pages/experiment-design.html, раздел 3.")

st.markdown(
    r"""
Размер выборки на группу для теста разницы пропорций (конверсий):

$$ n = \frac{\left(z_{1-\alpha/2} + z_{1-\beta}\right)^2 \cdot \left(p_1(1-p_1) + p_2(1-p_2)\right)}{(p_1 - p_2)^2} $$

где $p_1$ — базовая конверсия, $p_2 = p_1 + \text{MDE}$, $z_{1-\alpha/2}$ и $z_{1-\beta}$ — квантили
нормального распределения для уровня значимости и мощности.
"""
)

st.header("Калькулятор размера выборки")

col1, col2 = st.columns(2)
with col1:
    baseline = st.slider("Базовая конверсия p₁", 0.01, 0.50, 0.10, 0.01)
    mde_abs = st.slider("MDE (абсолютная разница, п.п.)", 0.001, 0.10, 0.02, 0.001, format="%.3f")
with col2:
    alpha = st.slider("Уровень значимости α", 0.01, 0.20, 0.05, 0.01)
    power_target = st.slider("Требуемая мощность", 0.50, 0.99, 0.80, 0.01)

p1 = baseline
p2 = baseline + mde_abs
z_alpha = stats.norm.ppf(1 - alpha / 2)
z_beta = stats.norm.ppf(power_target)

n_required = ((z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))) / (mde_abs ** 2)
n_required = int(np.ceil(n_required))

st.markdown(
    f"""
**Требуемый размер выборки на каждую группу: {n_required:,}** (всего ≈ {2*n_required:,} наблюдений)

- $p_1$ (базовая конверсия) = {p1:.3f}
- $p_2$ (конверсия с эффектом) = {p2:.3f}
- $z_{{1-\\alpha/2}}$ = {z_alpha:.3f}
- $z_{{1-\\beta}}$ = {z_beta:.3f}
"""
)

st.header("Мощность теста в зависимости от размера выборки")

n_range = np.linspace(max(10, n_required * 0.05), n_required * 2.5, 200)


def compute_power(n, p1, p2, alpha):
    se = np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    delta = abs(p2 - p1)
    z_score = delta / se
    power = stats.norm.cdf(z_score - z_alpha) + stats.norm.cdf(-z_score - z_alpha)
    return power


powers = [compute_power(n, p1, p2, alpha) for n in n_range]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(n_range, powers, color="steelblue")
ax.axhline(power_target, color="gray", linestyle="--", label=f"целевая мощность = {power_target}")
ax.axvline(n_required, color="red", linestyle="--", label=f"требуемый n = {n_required}")
ax.set_xlabel("размер выборки на группу (n)")
ax.set_ylabel("мощность теста")
ax.set_ylim(0, 1.02)
ax.legend()
ax.set_title("Зависимость мощности от размера выборки")
st.pyplot(fig)

st.header("MDE при заданном размере выборки")

st.markdown(
    "Если у вас уже фиксирован размер выборки (например, объём доступного трафика за период), "
    "можно посчитать MDE — минимальный эффект, который тест способен обнаружить с заданной мощностью."
)

n_fixed = st.slider("Доступный размер выборки на группу", 100, 200000, 10000, 100)

# Решаем относительно MDE приблизительно через se при p1 (без эффекта)
se_fixed = np.sqrt(2 * p1 * (1 - p1) / n_fixed)
mde_fixed = (z_alpha + z_beta) * se_fixed

st.markdown(
    f"При $n$ = {n_fixed:,} на группу, базовой конверсии {p1:.3f}, $\\alpha$ = {alpha} и мощности "
    f"{power_target}, MDE ≈ **{mde_fixed:.4f}** (абсолютная разница) или "
    f"**{mde_fixed/p1:.1%}** относительного прироста."
)

st.header("Симуляция: распределение p-value по многим A/A и A/B тестам")

st.markdown(
    "Симулируем много экспериментов: либо без реального эффекта (A/A-тест, $p_2=p_1$), либо с "
    "заданным эффектом (A/B-тест, $p_2 = p_1 + \\text{MDE}$), используя выбранный выше размер выборки "
    f"$n$ = {n_required:,}, и смотрим, в какой доле случаев p-value < α."
)

n_sims = st.slider("Число симуляций", 100, 5000, 1000, 100)

if st.button("Запустить симуляцию A/A и A/B"):
    rng = np.random.default_rng(7)

    sig_aa = 0
    sig_ab = 0

    for _ in range(n_sims):
        # A/A
        conv_a1 = rng.binomial(n_required, p1)
        conv_a2 = rng.binomial(n_required, p1)
        _, p_aa = proportions_ztest([conv_a1, conv_a2], [n_required, n_required])

        # A/B
        conv_b1 = rng.binomial(n_required, p1)
        conv_b2 = rng.binomial(n_required, p2)
        _, p_ab = proportions_ztest([conv_b1, conv_b2], [n_required, n_required])

        if p_aa < alpha:
            sig_aa += 1
        if p_ab < alpha:
            sig_ab += 1

    col1, col2 = st.columns(2)
    with col1:
        st.metric("A/A: доля значимых результатов (≈ α)", f"{sig_aa/n_sims:.3f}")
    with col2:
        st.metric("A/B: доля значимых результатов (≈ мощность)", f"{sig_ab/n_sims:.3f}")

    st.markdown(
        f"Теоретически: A/A ≈ {alpha:.3f} (ложноположительные при отсутствии эффекта), "
        f"A/B ≈ {power_target:.3f} (мощность при наличии эффекта MDE = {mde_abs})."
    )
