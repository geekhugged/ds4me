import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

st.title("3. A/B-тестирование: дизайн эксперимента и гипотезы")
st.caption("Теория и формулы — pages/data-product.html, раздел 3.")

st.markdown(
    r"""
Наблюдаемый эффект (lift) между тестовой и контрольной группой:

$$ \text{Lift} = \frac{p_{\text{treatment}} - p_{\text{control}}}{p_{\text{control}}} $$

z-статистика теста разности пропорций:

$$ z = \frac{p_{\text{treatment}} - p_{\text{control}}}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_{\text{treatment}}} + \frac{1}{n_{\text{control}}}\right)}} $$

где $\hat{p}$ — объединённая (pooled) оценка конверсии по обеим группам.
"""
)

st.header("Конструктор гипотезы")

st.markdown("Заполните структуру гипотезы для своего эксперимента (помогает дисциплинировать дизайн теста):")

change = st.text_input("Что меняем?", "Добавить кнопку быстрой оплаты на экран товара")
metric = st.text_input("Метрика успеха (OEC)", "Конверсия в покупку")
mechanism = st.text_area(
    "Ожидаемый эффект и механизм — почему это должно сработать?",
    "Пользователи отказываются от покупки из-за лишних шагов оформления; быстрая оплата "
    "сокращает путь до покупки и снижает долю отказов на этапе чекаута.",
)
guardrails = st.text_input("Guardrail-метрики (не должны деградировать)", "Средний чек, доля возвратов")

st.markdown(
    f"""
**Сформулированная гипотеза:**

Если мы внедрим «{change}», то «{metric}» изменится, потому что: {mechanism}

При этом будем следить, чтобы не деградировали: {guardrails}.
"""
)

st.header("Расчёт результата уже проведённого теста")

col1, col2 = st.columns(2)
with col1:
    n_control = st.number_input("Размер контрольной группы", 100, 1_000_000, 10_000, 100)
    conv_control = st.number_input("Число конверсий в контрольной группе", 0, 1_000_000, 800, 10)
with col2:
    n_treatment = st.number_input("Размер тестовой группы", 100, 1_000_000, 10_000, 100)
    conv_treatment = st.number_input("Число конверсий в тестовой группе", 0, 1_000_000, 880, 10)

alpha = st.slider("Уровень значимости α", 0.01, 0.20, 0.05, 0.01)

p_control = conv_control / n_control
p_treatment = conv_treatment / n_treatment
p_pool = (conv_control + conv_treatment) / (n_control + n_treatment)

se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_control + 1 / n_treatment))
z_stat = (p_treatment - p_control) / se_pool if se_pool > 0 else 0.0
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

lift = (p_treatment - p_control) / p_control if p_control > 0 else float("nan")

se_diff = np.sqrt(p_control * (1 - p_control) / n_control + p_treatment * (1 - p_treatment) / n_treatment)
z_crit = stats.norm.ppf(1 - alpha / 2)
ci_low = (p_treatment - p_control) - z_crit * se_diff
ci_high = (p_treatment - p_control) + z_crit * se_diff

col1, col2, col3 = st.columns(3)
col1.metric("Конверсия control", f"{p_control:.2%}")
col2.metric("Конверсия treatment", f"{p_treatment:.2%}")
col3.metric("Lift", f"{lift:+.2%}")

st.markdown(
    f"""
- z-статистика: **{z_stat:.3f}**
- p-value (двусторонний тест): **{p_value:.4f}**
- 95% доверительный интервал разницы конверсий: **[{ci_low:+.4f}, {ci_high:+.4f}]**
"""
)

if p_value < alpha:
    st.success(
        f"p-value < α ({alpha}): результат статистически значим. Решение о внедрении должно "
        "также учитывать величину эффекта (бизнес-значимость) и состояние guardrail-метрик."
    )
else:
    st.info(
        f"p-value ≥ α ({alpha}): недостаточно доказательств, чтобы отвергнуть гипотезу об "
        "отсутствии эффекта. Это не доказывает, что эффекта нет — возможно, не хватило мощности "
        "теста (см. тему про мощность и размер выборки)."
    )

fig, ax = plt.subplots(figsize=(7, 4))
x = np.linspace(-4, 4, 400)
ax.plot(x, stats.norm.pdf(x), color="gray", label="распределение z при H0")
ax.axvline(z_stat, color="red", linestyle="--", label=f"наблюдаемое z = {z_stat:.2f}")
ax.axvline(z_crit, color="steelblue", linestyle=":", label=f"критическое z = ±{z_crit:.2f}")
ax.axvline(-z_crit, color="steelblue", linestyle=":")
ax.fill_between(x, 0, stats.norm.pdf(x), where=(np.abs(x) >= z_crit), color="steelblue", alpha=0.2)
ax.legend(fontsize=8)
ax.set_title("Положение наблюдаемой статистики относительно критической области")
st.pyplot(fig)

st.header("Риски дизайна: симуляция peeking (раннего подглядывания)")

st.markdown(
    "Симуляция A/A-теста (без реального эффекта), где мы проверяем p-value после каждой новой "
    "порции данных и останавливаемся, как только видим p < α — показывает, как часто такая "
    "практика приводит к ложному «значимому» результату по сравнению с фиксированным размером "
    "выборки."
)

n_sims = st.slider("Число симуляций A/A-теста", 100, 3000, 500, 100)
n_checks = st.slider("Число промежуточных проверок p-value за тест", 1, 20, 10, 1)
n_final = st.slider("Финальный размер выборки на группу", 500, 20000, 5000, 500)
p_true = st.slider("Истинная конверсия (одинаковая в обеих группах, A/A)", 0.01, 0.5, 0.1, 0.01)

if st.button("Запустить симуляцию peeking"):
    rng = np.random.default_rng(0)
    false_positive_peeking = 0
    false_positive_fixed = 0
    check_points = np.linspace(n_final // n_checks, n_final, n_checks).astype(int)

    for _ in range(n_sims):
        conv_a = rng.binomial(1, p_true, n_final)
        conv_b = rng.binomial(1, p_true, n_final)

        stopped_early = False
        for cp in check_points:
            ca, cb = conv_a[:cp].sum(), conv_b[:cp].sum()
            pa, pb = ca / cp, cb / cp
            pp = (ca + cb) / (2 * cp)
            se = np.sqrt(pp * (1 - pp) * (2 / cp)) if pp > 0 else 1e-9
            z = (pb - pa) / se if se > 0 else 0
            pv = 2 * (1 - stats.norm.cdf(abs(z)))
            if pv < alpha:
                stopped_early = True
                break
        if stopped_early:
            false_positive_peeking += 1

        ca_final, cb_final = conv_a.sum(), conv_b.sum()
        pp_final = (ca_final + cb_final) / (2 * n_final)
        se_final = np.sqrt(pp_final * (1 - pp_final) * (2 / n_final)) if pp_final > 0 else 1e-9
        z_final = (cb_final / n_final - ca_final / n_final) / se_final
        pv_final = 2 * (1 - stats.norm.cdf(abs(z_final)))
        if pv_final < alpha:
            false_positive_fixed += 1

    col1, col2 = st.columns(2)
    col1.metric("Ложноположительные при peeking", f"{false_positive_peeking/n_sims:.1%}")
    col2.metric("Ложноположительные при фиксированной выборке", f"{false_positive_fixed/n_sims:.1%}")
    st.markdown(
        f"Ожидаемая доля ложноположительных при фиксированной выборке ≈ α = {alpha:.0%}. "
        "При peeking с несколькими проверками доля ложноположительных систематически выше — "
        "это и есть статистическая цена раннего подглядывания за результатами."
    )
