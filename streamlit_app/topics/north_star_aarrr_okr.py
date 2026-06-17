import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("1. North Star Metric, AARRR и OKR")
st.caption("Теория и формулы — pages/data-product.html, раздел 1.")

st.markdown(
    r"""
North Star Metric часто раскладывают на input-метрики (драйверы роста). Один из типичных вариантов
разложения:

$$ \text{NSM} = \text{DAU} \times \text{Activation rate} \times \text{Avg. value per active user} $$

Изменяя каждый из трёх множителей, можно увидеть, какой рычаг (трафик, активация или глубина
использования) даёт наибольший вклад в итоговую North Star Metric.
"""
)

st.header("Калькулятор North Star Metric: сравнение рычагов роста")

col1, col2, col3 = st.columns(3)
with col1:
    dau = st.slider("DAU (активных пользователей в день)", 1000, 1_000_000, 100_000, 1000)
with col2:
    activation = st.slider("Activation rate (доля дошедших до ценности)", 0.01, 1.0, 0.35, 0.01)
with col3:
    value_per_user = st.slider("Среднее значение ценности на активированного пользователя", 0.1, 10.0, 1.5, 0.1)

nsm = dau * activation * value_per_user
st.metric("North Star Metric", f"{nsm:,.0f}")

st.markdown("Сравним, как изменение каждого рычага на одинаковый относительный процент влияет на NSM:")

delta_pct = st.slider("Относительное улучшение каждого рычага по отдельности, %", 1, 50, 10, 1) / 100.0

nsm_dau_up = dau * (1 + delta_pct) * activation * value_per_user
nsm_act_up = dau * activation * (1 + delta_pct) * value_per_user
nsm_val_up = dau * activation * value_per_user * (1 + delta_pct)

fig, ax = plt.subplots(figsize=(7, 4))
levers = ["Базовый NSM", f"+{delta_pct:.0%} DAU", f"+{delta_pct:.0%} Activation", f"+{delta_pct:.0%} Value/user"]
values = [nsm, nsm_dau_up, nsm_act_up, nsm_val_up]
colors = ["gray", "steelblue", "seagreen", "indianred"]
ax.bar(levers, values, color=colors)
ax.set_ylabel("North Star Metric")
ax.set_title("Сравнение рычагов роста (при равном относительном улучшении)")
for i, v in enumerate(values):
    ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
st.pyplot(fig)

st.markdown(
    "Поскольку NSM в этой модели — произведение трёх множителей, относительный прирост NSM "
    "одинаков независимо от того, какой из множителей улучшить на тот же процент (с точностью "
    "до округления). На практике рычаги асимметричны по стоимости: повысить DAU часто значит "
    "тратить больше на маркетинг (рост CAC), а повысить Activation rate или Value per user — "
    "это, как правило, продуктовые улучшения, которые масштабируются без роста переменных затрат."
)

st.header("AARRR: воронка жизненного цикла пользователя")

st.markdown(
    "Задайте конверсию между стадиями AARRR, чтобы увидеть, где сильнее всего «протекает» воронка."
)

n_acquired = st.slider("Acquisition: число привлечённых пользователей", 100, 100000, 10000, 100)
cr_activation = st.slider("Acquisition → Activation, конверсия", 0.05, 1.0, 0.4, 0.01)
cr_retention = st.slider("Activation → Retention (вернулся через неделю), конверсия", 0.05, 1.0, 0.3, 0.01)
cr_revenue = st.slider("Retention → Revenue (стал платящим), конверсия", 0.01, 1.0, 0.15, 0.01)
cr_referral = st.slider("Revenue → Referral (порекомендовал), конверсия", 0.01, 1.0, 0.1, 0.01)

stages = ["Acquisition", "Activation", "Retention", "Revenue", "Referral"]
counts = [n_acquired]
for cr in [cr_activation, cr_retention, cr_revenue, cr_referral]:
    counts.append(counts[-1] * cr)

fig2, ax2 = plt.subplots(figsize=(8, 4.5))
ax2.bar(stages, counts, color="slateblue")
for i, c in enumerate(counts):
    ax2.text(i, c, f"{c:,.0f}", ha="center", va="bottom", fontsize=9)
ax2.set_ylabel("число пользователей")
ax2.set_title("Воронка AARRR")
st.pyplot(fig2)

drop_offs = [(stages[i], stages[i + 1], 1 - counts[i + 1] / counts[i]) for i in range(len(stages) - 1)]
worst = max(drop_offs, key=lambda x: x[2])
st.markdown(
    f"Наибольшая потеря пользователей происходит на переходе **{worst[0]} → {worst[1]}**: "
    f"теряется **{worst[2]:.1%}** пользователей. Это стадия, на которую разумно направить "
    "усилия команды в первую очередь."
)

st.header("OKR: декомпозиция цели")

st.markdown(
    """
OKR не считается формулой, но полезно тренироваться различать качественную цель (Objective) и
измеримые результаты (Key Results). Ниже — пример хорошей и плохой формулировки.
"""
)

example = st.radio(
    "Выберите пример для разбора",
    ["Хороший OKR", "Плохой OKR (задачи вместо результатов)"],
    horizontal=True,
)

if example == "Хороший OKR":
    st.success(
        "**Objective:** Стать самым удобным сервисом для быстрых платежей в регионе.\n\n"
        "**KR1:** Снизить долю неуспешных платежей с 8% до 3%.\n\n"
        "**KR2:** Повысить долю пользователей, совершающих повторный платёж в течение 30 дней, "
        "с 40% до 55%.\n\n"
        "**KR3:** Снизить среднее время оформления платежа с 45 до 20 секунд."
    )
else:
    st.error(
        "**Objective:** Улучшить платежи.\n\n"
        "**KR1:** Запустить новый экран оплаты.\n\n"
        "**KR2:** Интегрировать 2 новых платёжных провайдера.\n\n"
        "**KR3:** Провести редизайн формы оплаты.\n\n"
        "Проблема: все три KR — это задачи (можно выполнить и отчитаться), а не измеримые "
        "результаты. Их можно сделать на 100% и при этом не получить ни одного бизнес-эффекта."
    )
