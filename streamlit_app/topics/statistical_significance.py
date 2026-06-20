import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

st.title("20. Статистическая значимость, p-value и доверительные интервалы")
st.caption("Теория и формулы — pages/ml-theory.html, раздел 20.")

st.markdown(
    r"""
В ML результат измерения (метрика модели A против модели B, конверсия в A/B-тесте) —
случайная величина. Чтобы отличить реальный эффект от шума, проверяют нулевую гипотезу
$H_0$: «истинной разницы нет».

- **p-value** — вероятность получить наблюдаемый (или более экстремальный) результат при
  верной $H_0$. Если $p < \alpha$, эффект считают значимым.
- **Доверительный интервал** уровня $1-\alpha$ для разности долей:
  $$ (\hat{p}_B - \hat{p}_A) \pm z_{1-\alpha/2}\sqrt{\frac{\hat{p}_A(1-\hat{p}_A)}{n_A} + \frac{\hat{p}_B(1-\hat{p}_B)}{n_B}}. $$
- **Ошибка I рода** ($\alpha$) — ложное обнаружение эффекта; **ошибка II рода** ($\beta$) —
  пропуск реального эффекта; **мощность** $= 1-\beta$.

Здесь $\hat{p}_A, \hat{p}_B$ — наблюдаемые доли успеха, $n_A, n_B$ — размеры выборок,
$z_{1-\alpha/2}$ — квантиль стандартного нормального распределения.
"""
)

st.header("Сравнение двух моделей / вариантов по доле успеха")

st.markdown(
    "Сценарий: вариант A (baseline) и вариант B (новая модель) дают бинарный исход "
    "(клик/правильный ответ). Проверяем, значимо ли B лучше A (z-тест для двух долей)."
)

p_a = st.slider("Истинная доля успеха A (baseline)", 0.05, 0.95, 0.30, 0.01)
true_lift = st.slider("Истинная разница B − A", -0.20, 0.20, 0.05, 0.01)
n_per_group = st.slider("Размер выборки на группу", 50, 5000, 1000, 50)
alpha = st.select_slider("Уровень значимости α", options=[0.01, 0.05, 0.10], value=0.05)

p_b = float(np.clip(p_a + true_lift, 0.001, 0.999))

rng = np.random.default_rng(7)
succ_a = rng.binomial(n_per_group, p_a)
succ_b = rng.binomial(n_per_group, p_b)
phat_a = succ_a / n_per_group
phat_b = succ_b / n_per_group
diff = phat_b - phat_a

# z-тест для двух долей (pooled)
p_pool = (succ_a + succ_b) / (2 * n_per_group)
se_pool = np.sqrt(p_pool * (1 - p_pool) * (2 / n_per_group))
z_stat = diff / se_pool if se_pool > 0 else 0.0
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

z_crit = stats.norm.ppf(1 - alpha / 2)
se_ci = np.sqrt(phat_a * (1 - phat_a) / n_per_group + phat_b * (1 - phat_b) / n_per_group)
ci_low = diff - z_crit * se_ci
ci_high = diff + z_crit * se_ci

significant = p_value < alpha

col1, col2, col3 = st.columns(3)
col1.metric("Наблюдаемая разница B−A", f"{diff:+.3f}")
col2.metric("z-статистика", f"{z_stat:.2f}")
col3.metric("p-value", f"{p_value:.4f}", "значимо" if significant else "не значимо")

st.write(
    f"{int((1 - alpha) * 100)}% доверительный интервал для разности: "
    f"**[{ci_low:+.3f}, {ci_high:+.3f}]**. "
    + ("Не содержит 0 — разница значима." if ci_low > 0 or ci_high < 0 else "Содержит 0 — разница не значима.")
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

# Распределение z-статистики при H0 и зоны отвержения
xs = np.linspace(-4, 4, 400)
axes[0].plot(xs, stats.norm.pdf(xs), color="black", label="распределение z при $H_0$")
axes[0].fill_between(xs, 0, stats.norm.pdf(xs), where=(np.abs(xs) >= z_crit), color="salmon", alpha=0.6, label="зона отвержения")
axes[0].axvline(z_stat, color="blue", lw=2, label=f"наблюдённое z={z_stat:.2f}")
axes[0].set_title("z-статистика и критические зоны")
axes[0].set_xlabel("z")
axes[0].legend(fontsize=8)

# Доверительный интервал
axes[1].errorbar([diff], [0], xerr=[[diff - ci_low], [ci_high - diff]], fmt="o", color="blue", capsize=6, lw=2)
axes[1].axvline(0, color="gray", ls="--", label="нет эффекта (0)")
axes[1].scatter([true_lift], [0.15], color="green", marker="v", s=80, label="истинная разница")
axes[1].set_yticks([])
axes[1].set_xlabel("разность долей B − A")
axes[1].set_title(f"{int((1 - alpha) * 100)}% доверительный интервал")
axes[1].legend(fontsize=8)
st.pyplot(fig)

st.markdown("**Анализ ошибок I/II рода при заданных параметрах** (через нормальное приближение):")

# Мощность теста при истинной разнице true_lift и шаге alpha
se0 = np.sqrt(2 * p_a * (1 - p_a) / n_per_group)
se1 = np.sqrt(p_a * (1 - p_a) / n_per_group + p_b * (1 - p_b) / n_per_group)
if true_lift != 0 and se1 > 0:
    z_a = stats.norm.ppf(1 - alpha / 2)
    power = stats.norm.cdf((abs(true_lift) - z_a * se0) / se1) + stats.norm.cdf((-abs(true_lift) - z_a * se0) / se1)
    power = float(np.clip(power, 0, 1))
else:
    power = alpha  # при нулевом эффекте «мощность» = вероятность ложного срабатывания

c1, c2 = st.columns(2)
c1.metric("α (ошибка I рода)", f"{alpha:.2f}")
c2.metric("Мощность 1−β" if true_lift != 0 else "P(отвергнуть H0 | эффекта нет)", f"{power:.3f}")

st.markdown(
    """
Увеличивайте размер выборки — доверительный интервал сужается, и даже небольшая истинная
разница становится статистически значимой (растёт мощность). При нулевой истинной разнице
тест всё равно «находит» эффект примерно в α долях случаев — это и есть ошибка I рода.

Типичные ловушки в ML: подглядывание в тест (peeking) и множественные сравнения завышают
вероятность ложного открытия; статистическая значимость не равна практической значимости —
смотрите на размер эффекта и доверительный интервал, а не только на p-value.
"""
)
