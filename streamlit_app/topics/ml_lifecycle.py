import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("1. Жизненный цикл ML-проекта")
st.caption("Теория — pages/mlops.html, раздел 1.")

st.markdown(
    r"""
Жизненный цикл ML-проекта — итеративный процесс: постановка задачи → данные → эксперименты →
валидация → деплой → мониторинг → переобучение → (возможно) decommissioning.

Один из главных рисков ранних стадий — улучшать **ML-метрику** (например, AUC, MAE), не
проверив, переводится ли это улучшение в **бизнес-метрику** (revenue, retention, конверсия).
Связь между ними можно приблизить линейно:

$$ \Delta \text{Business} \approx \frac{\partial \text{Business}}{\partial \text{ML metric}} \cdot \Delta \text{ML metric} $$

Симуляция ниже показывает, как разная "чувствительность" бизнес-метрики к ML-метрике (а также
её неопределённость) влияет на то, оправдывают ли инвестиции в улучшение модели ожидаемый
бизнес-эффект.
"""
)

st.header("Симуляция: окупаемость улучшения модели")

col1, col2 = st.columns(2)
with col1:
    delta_ml = st.slider(
        "Ожидаемое улучшение ML-метрики (например, +AUC)", 0.0, 0.10, 0.02, 0.005,
        format="%.3f",
    )
    sensitivity = st.slider(
        "Чувствительность бизнес-метрики к ML-метрике (∂Business/∂ML)",
        0.0, 500_000.0, 100_000.0, 5_000.0,
        help="Например, во сколько $ выручки в год переводится +0.01 AUC",
    )
with col2:
    sensitivity_uncertainty = st.slider(
        "Неопределённость чувствительности (стандартное отклонение, %)", 0, 100, 30, 5,
    )
    project_cost = st.slider(
        "Стоимость проекта (разработка + инфраструктура), $", 0, 1_000_000, 150_000, 10_000,
    )

n_sims = 5000
rng = np.random.default_rng(7)

sens_samples = rng.normal(
    loc=sensitivity, scale=sensitivity * (sensitivity_uncertainty / 100), size=n_sims
)
sens_samples = np.clip(sens_samples, 0, None)

business_gain = sens_samples * delta_ml

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(business_gain, bins=50, color="steelblue", alpha=0.8)
ax.axvline(project_cost, color="crimson", linestyle="--", label=f"стоимость проекта = ${project_cost:,.0f}")
ax.axvline(business_gain.mean(), color="black", linestyle="-", label=f"среднее ожид. выгода = ${business_gain.mean():,.0f}")
ax.set_xlabel("ожидаемая годовая бизнес-выгода, $")
ax.set_ylabel("частота (из симуляций)")
ax.set_title("Распределение ожидаемой бизнес-выгоды от улучшения модели")
ax.legend()
st.pyplot(fig)

prob_profitable = (business_gain > project_cost).mean()

st.markdown(
    f"""
**Средняя ожидаемая бизнес-выгода:** ${business_gain.mean():,.0f} в год
(при улучшении ML-метрики на {delta_ml:.3f}).

**Вероятность, что выгода превысит стоимость проекта:** {prob_profitable:.1%}
"""
)

if prob_profitable < 0.5:
    st.warning(
        "При текущих предположениях улучшение модели с вероятностью больше 50% **не окупит** "
        "стоимость проекта. Прежде чем инвестировать в моделирование, стоит уточнить связь "
        "ML-метрики с бизнес-метрикой (например, через небольшой A/B-тест на текущей модели)."
    )
else:
    st.success(
        "При текущих предположениях улучшение модели с вероятностью больше 50% **окупит** "
        "стоимость проекта. Но обратите внимание на ширину распределения — высокая "
        "неопределённость чувствительности означает высокий риск."
    )

st.header("Стадии жизненного цикла и обратные связи")

st.markdown(
    """
Жизненный цикл не линеен: мониторинг продакшена постоянно поставляет сигналы, которые
возвращают команду к более ранним стадиям. Ниже — упрощённая схема стадий и типичных
"обратных переходов".
"""
)

stages = [
    "Постановка задачи",
    "Сбор и разметка данных",
    "EDA / feature engineering",
    "Эксперименты и обучение",
    "Валидация перед релизом",
    "Деплой",
    "Мониторинг",
    "Переобучение / итерации",
]

fig2, ax2 = plt.subplots(figsize=(9, 3.5))
x = np.arange(len(stages))
ax2.plot(x, np.zeros_like(x), "o-", color="steelblue", markersize=10)
for i, s in enumerate(stages):
    ax2.annotate(s, (i, 0), textcoords="offset points", xytext=(0, 12 if i % 2 == 0 else -25),
                 ha="center", fontsize=8, rotation=0)

# обратные связи
ax2.annotate("", xy=(2, 0.02), xytext=(6, 0.02),
              arrowprops=dict(arrowstyle="->", color="crimson", connectionstyle="arc3,rad=-0.4"))
ax2.text(4, 0.18, "дрифт данных → новый EDA", color="crimson", ha="center", fontsize=8)

ax2.annotate("", xy=(3, -0.02), xytext=(7, -0.02),
              arrowprops=dict(arrowstyle="->", color="darkorange", connectionstyle="arc3,rad=0.4"))
ax2.text(5, -0.22, "деградация метрик → переобучение", color="darkorange", ha="center", fontsize=8)

ax2.set_xlim(-0.5, len(stages) - 0.5)
ax2.set_ylim(-0.35, 0.35)
ax2.axis("off")
st.pyplot(fig2)
