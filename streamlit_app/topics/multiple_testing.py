import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats
from statsmodels.stats.multitest import multipletests

st.title("4. Множественное тестирование и поправки (Bonferroni, FDR/Benjamini-Hochberg)")
st.caption("Теория и формулы — pages/experiment-design.html, раздел 4.")

st.markdown(
    r"""
Если проверять $m$ независимых гипотез, каждую на уровне значимости $\alpha$, то вероятность
хотя бы одного ложноположительного результата (FWER) растёт как

$$ \text{FWER} = 1 - (1-\alpha)^m $$

**Поправка Бонферрони**: проверять каждую гипотезу на уровне $\alpha/m$.

**Процедура Бенджамини-Хохберга (BH)** контролирует FDR — ожидаемую долю ложных открытий среди
найденных значимых результатов — и при этом сохраняет больше мощности, чем Бонферрони.
"""
)

st.header("FWER в зависимости от числа тестов")

alpha = st.slider("Уровень значимости α на один тест", 0.01, 0.20, 0.05, 0.01)
m_max = st.slider("Максимальное число тестов m", 1, 100, 30, 1)

m_range = np.arange(1, m_max + 1)
fwer = 1 - (1 - alpha) ** m_range
fwer_bonf = np.minimum(1, m_range * (alpha / m_range))  # после поправки Бонферрони FWER ~ alpha

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(m_range, fwer, color="crimson", label="без поправки (FWER)")
ax.axhline(alpha, color="gray", linestyle="--", label=f"α = {alpha}")
ax.set_xlabel("число тестов (m)")
ax.set_ylabel("вероятность хотя бы одной ошибки I рода")
ax.set_ylim(0, 1.02)
ax.legend()
ax.set_title("Рост FWER без поправки на множественное тестирование")
st.pyplot(fig)

st.markdown(
    f"При m = {m_max} тестах без поправки вероятность хотя бы одного ложноположительного "
    f"результата ≈ **{fwer[-1]:.1%}**, даже если ни одного реального эффекта нет."
)

st.header("Симуляция: доля ложных открытий с поправкой и без")

st.markdown(
    "Симулируем $m$ тестов, из которых доля `true_effect_frac` имеют реальный эффект "
    "(под $H_1$), остальные — под $H_0$ (нет эффекта). Сравним три стратегии: без поправки, "
    "Bonferroni, Benjamini-Hochberg."
)

col1, col2, col3 = st.columns(3)
with col1:
    m_tests = st.slider("Число тестов (m)", 5, 200, 50, 5)
with col2:
    true_effect_frac = st.slider("Доля тестов с реальным эффектом", 0.0, 1.0, 0.2, 0.05)
with col3:
    effect_size = st.slider("Размер реального эффекта (Cohen's d)", 0.1, 2.0, 0.5, 0.1)

n_per_test = st.slider("Размер выборки на тест (на группу)", 10, 500, 50, 10)
fdr_q = st.slider("Целевой уровень FDR (q) для BH", 0.01, 0.20, 0.05, 0.01)

if st.button("Запустить симуляцию множественного тестирования"):
    rng = np.random.default_rng(11)

    n_true_effects = int(round(m_tests * true_effect_frac))
    is_true_effect = np.zeros(m_tests, dtype=bool)
    is_true_effect[:n_true_effects] = True
    rng.shuffle(is_true_effect)

    pvalues = np.zeros(m_tests)
    for i in range(m_tests):
        x1 = rng.normal(0, 1, size=n_per_test)
        mu2 = effect_size if is_true_effect[i] else 0.0
        x2 = rng.normal(mu2, 1, size=n_per_test)
        _, p = stats.ttest_ind(x1, x2)
        pvalues[i] = p

    # Без поправки
    reject_none = pvalues < alpha

    # Bonferroni
    reject_bonf, _, _, _ = multipletests(pvalues, alpha=alpha, method="bonferroni")[:4]

    # Benjamini-Hochberg
    reject_bh, _, _, _ = multipletests(pvalues, alpha=fdr_q, method="fdr_bh")[:4]

    def summarize(reject_mask, is_true_effect):
        n_rejected = reject_mask.sum()
        false_positives = (reject_mask & ~is_true_effect).sum()
        true_positives = (reject_mask & is_true_effect).sum()
        fdr = false_positives / n_rejected if n_rejected > 0 else 0.0
        power = true_positives / is_true_effect.sum() if is_true_effect.sum() > 0 else 0.0
        return n_rejected, false_positives, fdr, power

    results = {
        "Без поправки": summarize(reject_none, is_true_effect),
        "Bonferroni": summarize(reject_bonf, is_true_effect),
        f"Benjamini-Hochberg (q={fdr_q})": summarize(reject_bh, is_true_effect),
    }

    st.markdown("### Результаты")

    for name, (n_rej, fp, fdr_val, power_val) in results.items():
        st.markdown(
            f"**{name}**: отвергнуто {n_rej} из {m_tests} гипотез, из них ложных открытий "
            f"(false positives) = {fp}, наблюдаемый FDR = {fdr_val:.2f}, мощность = {power_val:.2f}"
        )

    # Визуализация p-values
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    sorted_idx = np.argsort(pvalues)
    sorted_p = pvalues[sorted_idx]
    colors = np.where(is_true_effect[sorted_idx], "seagreen", "gray")

    ax2.scatter(np.arange(1, m_tests + 1), sorted_p, c=colors, alpha=0.7,
                label="зелёный = реальный эффект, серый = H0 верна")
    ax2.axhline(alpha, color="crimson", linestyle="--", label=f"α = {alpha} (без поправки)")
    ax2.axhline(alpha / m_tests, color="orange", linestyle="--", label=f"α/m = {alpha/m_tests:.4f} (Bonferroni)")

    # BH линия: k/m * q
    k_range = np.arange(1, m_tests + 1)
    bh_line = (k_range / m_tests) * fdr_q
    ax2.plot(k_range, bh_line, color="purple", linestyle="--", label=f"k/m·q (BH, q={fdr_q})")

    ax2.set_xlabel("ранг p-value (по возрастанию)")
    ax2.set_ylabel("p-value")
    ax2.set_yscale("log")
    ax2.legend(fontsize=8)
    ax2.set_title("Отсортированные p-values и пороги коррекции")
    st.pyplot(fig2)

    st.markdown(
        "Точки ниже соответствующей линии порога — гипотезы, отвергаемые этим методом. "
        "BH-порог растёт линейно с рангом, что позволяет отвергнуть больше гипотез, чем "
        "фиксированный порог Bonferroni, сохраняя контроль над долей ложных открытий."
    )
