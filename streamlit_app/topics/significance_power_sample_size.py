import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

st.title("4. Статистическая значимость, p-value, мощность теста и размер выборки")
st.caption("Теория и формулы — pages/data-product.html, раздел 4.")

st.markdown(
    r"""
Размер выборки на группу для теста разницы двух пропорций:

$$ n = \frac{\left(z_{1-\alpha/2} + z_{1-\beta}\right)^2 \left(p_1(1-p_1) + p_2(1-p_2)\right)}{(p_1 - p_2)^2} $$

Мощность теста как функция размера выборки:

$$ \text{power} = \Phi\!\left(\frac{|p_1-p_2|}{\sqrt{\frac{p_1(1-p_1)}{n}+\frac{p_2(1-p_2)}{n}}} - z_{1-\alpha/2}\right) $$
"""
)

st.header("Интуиция p-value: распределение статистики под H0")

st.markdown(
    "p-value — это вероятность увидеть статистику настолько же (или более) экстремальную, как "
    "наблюдаемая, если на самом деле эффекта нет (H0 верна)."
)

z_observed = st.slider("Наблюдаемая z-статистика", -4.0, 4.0, 1.96, 0.05)
p_value_demo = 2 * (1 - stats.norm.cdf(abs(z_observed)))

fig, ax = plt.subplots(figsize=(7, 4))
x = np.linspace(-4.5, 4.5, 500)
ax.plot(x, stats.norm.pdf(x), color="black")
ax.fill_between(x, 0, stats.norm.pdf(x), where=(x >= abs(z_observed)), color="red", alpha=0.4)
ax.fill_between(x, 0, stats.norm.pdf(x), where=(x <= -abs(z_observed)), color="red", alpha=0.4, label="p-value (площадь)")
ax.axvline(z_observed, color="blue", linestyle="--", label=f"наблюдаемая z = {z_observed}")
ax.legend(fontsize=8)
ax.set_title(f"p-value (двусторонний) = {p_value_demo:.4f}")
st.pyplot(fig)

st.header("Ошибки I и II рода")

st.markdown(
    "Распределение тестовой статистики при отсутствии эффекта (H0) и при наличии эффекта (H1) "
    "пересекаются — отсюда два типа ошибок: ложноположительные (α, красная зона справа под H0) "
    "и ложноотрицательные (β, зона слева от критической границы под H1)."
)

effect_size_d = st.slider("Размер эффекта (стандартизованный, в σ)", 0.0, 3.0, 1.0, 0.1)
alpha_demo = st.slider("Уровень значимости α", 0.01, 0.20, 0.05, 0.01, key="alpha_demo")

z_crit_demo = stats.norm.ppf(1 - alpha_demo)
power_demo = 1 - stats.norm.cdf(z_crit_demo - effect_size_d)
beta_demo = 1 - power_demo

fig2, ax2 = plt.subplots(figsize=(8, 4.5))
x2 = np.linspace(-4, 4 + effect_size_d, 600)
h0_pdf = stats.norm.pdf(x2, 0, 1)
h1_pdf = stats.norm.pdf(x2, effect_size_d, 1)
ax2.plot(x2, h0_pdf, color="steelblue", label="распределение под H0 (эффекта нет)")
ax2.plot(x2, h1_pdf, color="seagreen", label="распределение под H1 (эффект есть)")
ax2.axvline(z_crit_demo, color="red", linestyle="--", label=f"критическая граница ({z_crit_demo:.2f})")
ax2.fill_between(x2, 0, h0_pdf, where=(x2 >= z_crit_demo), color="red", alpha=0.4, label="α (ложноположительные)")
ax2.fill_between(x2, 0, h1_pdf, where=(x2 <= z_crit_demo), color="orange", alpha=0.4, label="β (ложноотрицательные)")
ax2.legend(fontsize=8, loc="upper right")
ax2.set_title(f"α = {alpha_demo:.3f}, β = {beta_demo:.3f}, мощность (1-β) = {power_demo:.3f}")
st.pyplot(fig2)

st.header("Калькулятор размера выборки")

col1, col2 = st.columns(2)
with col1:
    baseline = st.slider("Базовая конверсия p₁", 0.01, 0.50, 0.10, 0.01)
    mde_abs = st.slider("MDE (абсолютная разница, п.п.)", 0.001, 0.10, 0.02, 0.001, format="%.3f")
with col2:
    alpha = st.slider("Уровень значимости α", 0.01, 0.20, 0.05, 0.01, key="alpha_calc")
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
"""
)

st.header("Как размер выборки растёт при уменьшении MDE")

mde_range = np.linspace(mde_abs * 0.3, mde_abs * 3, 100)
n_for_mde = []
for m in mde_range:
    p2_m = p1 + m
    n_m = ((z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2_m * (1 - p2_m))) / (m ** 2)
    n_for_mde.append(n_m)

fig3, ax3 = plt.subplots(figsize=(7, 4))
ax3.plot(mde_range, n_for_mde, color="purple")
ax3.axvline(mde_abs, color="gray", linestyle="--", label=f"выбранный MDE = {mde_abs}")
ax3.set_xlabel("MDE (абсолютная разница конверсий)")
ax3.set_ylabel("требуемый размер выборки на группу")
ax3.set_yscale("log")
ax3.legend()
ax3.set_title("Нелинейная зависимость n от MDE (n ~ 1/MDE²)")
st.pyplot(fig3)

st.markdown(
    "Зависимость размера выборки от MDE квадратичная и обратная: чтобы обнаружить эффект в 2 "
    "раза меньшего размера, нужно примерно в 4 раза больше данных."
)
