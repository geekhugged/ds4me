import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("7.4 Детекция фрода в платёжных транзакциях")
st.caption(
    "Кейс — pages/ml-system-design-interview.html, раздел 4 "
    "(«Детекция фрода в платёжных транзакциях»)."
)

st.markdown(
    r"""
Фрод-детекция — задача бинарной классификации с крайне несбалансированными классами (фрод
обычно < 0.1-1% транзакций). Из-за этого ROC-AUC может быть обманчиво высоким, а ключевые
метрики — **Precision-Recall AUC**, $F_\beta$-score (с приоритетом recall, $\beta > 1$) и
выбор порога через **cost-matrix**:

$$ E[\text{cost}] = C_{FP} \cdot P(FP) + C_{FN} \cdot P(FN) $$

где $C_{FP}$ — стоимость блокировки легитимной транзакции (потеря выручки + UX),
$C_{FN}$ — стоимость пропущенного фрода (прямой финансовый убыток + chargeback fees).

Ниже — симуляция: генерируем выборку транзакций с заданной долей фрода и качеством модели,
выбираем порог классификации и смотрим на precision/recall, $F_\beta$ и ожидаемую стоимость.
"""
)

st.header("Симуляция: выбор порога по cost-matrix")

col1, col2, col3 = st.columns(3)
with col1:
    fraud_rate = st.slider("Доля фрода среди транзакций", 0.001, 0.05, 0.005, 0.001, format="%.3f")
with col2:
    model_quality = st.slider(
        "Качество модели (separation)", 0.5, 5.0, 2.5, 0.1,
        help="Насколько сильно различаются распределения скоров фрода и легитимных транзакций",
    )
with col3:
    n_transactions = st.slider("Число транзакций в симуляции", 5_000, 200_000, 50_000, 5_000)

rng = np.random.default_rng(123)
n_fraud = int(n_transactions * fraud_rate)
n_legit = n_transactions - n_fraud

# скоры модели: легитимные ~ N(0,1), фрод ~ N(model_quality, 1), затем сигмоида в [0,1]
legit_scores = rng.normal(0, 1, n_legit)
fraud_scores = rng.normal(model_quality, 1, n_fraud)

scores = np.concatenate([legit_scores, fraud_scores])
labels = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])

probs = 1 / (1 + np.exp(-scores))

st.subheader("Cost-matrix")
col1, col2 = st.columns(2)
with col1:
    cost_fp = st.slider("Стоимость false positive (C_FP), $", 1, 200, 20)
with col2:
    cost_fn = st.slider("Стоимость false negative (C_FN), $", 10, 5000, 500, 10)

threshold = st.slider("Порог классификации (allow/block)", 0.0, 1.0, 0.5, 0.01)

preds = (probs >= threshold).astype(int)
tp = np.sum((preds == 1) & (labels == 1))
fp = np.sum((preds == 1) & (labels == 0))
fn = np.sum((preds == 0) & (labels == 1))
tn = np.sum((preds == 0) & (labels == 0))

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
beta = st.slider("β для F_beta (приоритет recall при β>1)", 0.5, 3.0, 2.0, 0.1)
f_beta = (
    (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    if (precision + recall) > 0
    else 0.0
)

expected_cost = cost_fp * fp + cost_fn * fn
total_cost_per_txn = expected_cost / n_transactions

m1, m2, m3, m4 = st.columns(4)
m1.metric("Precision", f"{precision:.3f}")
m2.metric("Recall", f"{recall:.3f}")
m3.metric(f"F_{beta:.1f}", f"{f_beta:.3f}")
m4.metric("Ожидаемая стоимость / транзакцию", f"${total_cost_per_txn:.4f}")

st.markdown(
    f"При выбранном пороге: TP={tp}, FP={fp}, FN={fn}, TN={tn}. "
    f"Суммарная ожидаемая стоимость на {n_transactions:,} транзакций: **${expected_cost:,.0f}**."
)

st.header("Оптимальный порог по cost-matrix")

thresholds = np.linspace(0.01, 0.99, 99)
costs = []
precisions = []
recalls = []
for t in thresholds:
    p = (probs >= t).astype(int)
    tp_ = np.sum((p == 1) & (labels == 1))
    fp_ = np.sum((p == 1) & (labels == 0))
    fn_ = np.sum((p == 0) & (labels == 1))
    costs.append(cost_fp * fp_ + cost_fn * fn_)
    prec = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0.0
    rec = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0.0
    precisions.append(prec)
    recalls.append(rec)

costs = np.array(costs)
best_idx = np.argmin(costs)
best_threshold = thresholds[best_idx]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1.plot(thresholds, costs, color="darkred")
ax1.axvline(best_threshold, color="green", linestyle="--", label=f"оптимальный порог = {best_threshold:.2f}")
ax1.axvline(threshold, color="blue", linestyle=":", label=f"выбранный порог = {threshold:.2f}")
ax1.set_xlabel("порог")
ax1.set_ylabel("ожидаемая стоимость, $")
ax1.set_title("Ожидаемая стоимость vs порог")
ax1.legend()

ax2.plot(recalls, precisions, color="steelblue")
idx_sel = np.argmin(np.abs(thresholds - threshold))
idx_best = best_idx
ax2.scatter([recalls[idx_sel]], [precisions[idx_sel]], color="blue", label=f"выбранный порог", zorder=5)
ax2.scatter([recalls[idx_best]], [precisions[idx_best]], color="green", label="оптимальный порог", zorder=5)
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall curve")
ax2.legend()

st.pyplot(fig)

st.metric(
    "Оптимальный порог (минимизирует ожидаемую стоимость)",
    f"{best_threshold:.2f}",
    delta=f"экономия ${costs[idx_sel] - costs[best_idx]:,.0f} vs выбранный порог" if costs[idx_sel] != costs[best_idx] else None,
)

st.markdown(
    """
**На что обратить внимание:**
- При высоком `C_FN` (дорогой пропущенный фрод) оптимальный порог сдвигается влево
  (модель блокирует/проверяет больше транзакций, recall растёт за счёт precision).
- При высоком `C_FP` (дорогая блокировка легитимных платежей — например, потеря крупного
  клиента) оптимальный порог сдвигается вправо.
- Низкий `fraud_rate` (сильный имбаланс) делает precision крайне чувствительным к порогу —
  даже небольшое смещение порога резко меняет число false positives относительно числа фрода.
- На практике пороги обычно различаются по сегментам (например, более консервативный порог для
  high-value транзакций, где $C_{FN}$ выше).
"""
)
