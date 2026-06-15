import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("7.2 Предсказание CTR для рекламного аукциона")
st.caption(
    "Кейс — pages/ml-system-design-interview.html, раздел 2 "
    "(«Предсказание CTR для рекламного аукциона»)."
)

st.markdown(
    r"""
В рекламном аукционе объявления ранжируются по $\text{rank\_score} = pCTR \times \text{bid}$, поэтому
$pCTR$ должен быть не просто "ранжирующим" (высокий AUC), но и хорошо **калиброванным**:
$E[\hat{p} \mid \hat{p} = p] \approx p$.

Типичная проблема: для борьбы с имбалансом классов (CTR обычно < 1-5%) при обучении применяют
**downsampling** негативного класса с коэффициентом $w$. Это ускоряет обучение, но смещает
предсказанные вероятности — модель "думает", что положительного класса больше, чем на самом
деле. Корректировка калибровки:

$$ q_{calibrated} = \frac{q}{q + (1-q)/w} $$

где $q$ — "сырое" предсказание модели, обученной на downsampled данных, $w$ — коэффициент
downsampling (например, $w=0.1$ означает, что оставлен только каждый 10-й негативный пример).

Ниже — симуляция: смотрим, как downsampling искажает калибровку и как формула выше восстанавливает
реальный $pCTR$.
"""
)

st.header("Симуляция калибровки при downsampling негативов")

col1, col2, col3 = st.columns(3)
with col1:
    true_ctr = st.slider("Истинный средний CTR", 0.005, 0.10, 0.02, 0.005, format="%.3f")
with col2:
    downsample_w = st.slider(
        "Коэффициент downsampling негативов (w)", 0.05, 1.0, 0.2, 0.05,
        help="w=1 — без downsampling, w=0.1 — оставлен каждый 10-й негатив",
    )
with col3:
    n_samples = st.slider("Число показов в симуляции", 1_000, 200_000, 50_000, 1_000)

rng = np.random.default_rng(0)

# "Истинная" вероятность клика — варьируется по объявлениям (heterogeneous CTR)
true_p = np.clip(rng.beta(2, 2, n_samples) * true_ctr * 3, 0.0005, 0.5)
clicks = rng.binomial(1, true_p)

# downsampling негативов
neg_mask = clicks == 0
keep_neg = rng.random(n_samples) < downsample_w
keep_mask = clicks.astype(bool) | (neg_mask & keep_neg)

p_downsampled = true_p[keep_mask]
y_downsampled = clicks[keep_mask]

# "обученная" модель на downsampled данных в среднем выдаёт q ~ true_p / (true_p + (1-true_p)*w)
# (это смещённая оценка из-за downsampling) - вычислим напрямую как proxy
q_raw = true_p / (true_p + (1 - true_p) * downsample_w)
q_raw = np.clip(q_raw, 1e-6, 1 - 1e-6)

# применяем формулу коррекции
q_calibrated = q_raw / (q_raw + (1 - q_raw) / downsample_w)

# Бьём на бакеты по q_raw для построения reliability diagram
n_bins = 10
bins = np.linspace(0, max(q_raw.max(), q_calibrated.max(), true_p.max()) * 1.05, n_bins + 1)


def reliability(pred, target_true_p, bins):
    bin_idx = np.digitize(pred, bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(bins) - 2)
    mean_pred = []
    mean_true = []
    for b in range(len(bins) - 1):
        mask = bin_idx == b
        if mask.sum() > 0:
            mean_pred.append(pred[mask].mean())
            mean_true.append(target_true_p[mask].mean())
    return np.array(mean_pred), np.array(mean_true)


pred_raw_bins, true_bins_raw = reliability(q_raw, true_p, bins)
pred_cal_bins, true_bins_cal = reliability(q_calibrated, true_p, bins)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot([0, bins.max()], [0, bins.max()], "k--", label="идеальная калибровка")
ax.plot(pred_raw_bins, true_bins_raw, "o-", color="red", label="без коррекции (q_raw)")
ax.plot(pred_cal_bins, true_bins_cal, "o-", color="green", label="после коррекции (q_calibrated)")
ax.set_xlabel("предсказанная вероятность")
ax.set_ylabel("реальная (истинная) вероятность клика")
ax.set_title("Reliability diagram: эффект коррекции downsampling")
ax.legend()
st.pyplot(fig)

ece_raw = np.mean(np.abs(pred_raw_bins - true_bins_raw))
ece_cal = np.mean(np.abs(pred_cal_bins - true_bins_cal))

col1, col2 = st.columns(2)
col1.metric("Calibration error без коррекции", f"{ece_raw:.4f}")
col2.metric("Calibration error после коррекции", f"{ece_cal:.4f}")

st.markdown(
    """
**Вывод:** downsampling негативов искажает предсказанную вероятность в сторону переоценки CTR
(модель "видит" более высокую долю положительных примеров, чем в реальности). Формула коррекции
восстанавливает калибровку без необходимости переобучать модель на полных данных.
"""
)

st.header("Precision/Recall при разных порогах (имбаланс классов)")

st.markdown(
    "В задачах с редким положительным классом (как pCTR) выбор порога классификации сильно "
    "влияет на precision/recall. Ниже — кривая precision-recall на симулированных данных."
)

threshold = st.slider("Порог классификации (для иллюстрации precision/recall)", 0.0, 0.3, 0.02, 0.005)

# precision/recall по сетке порогов на исходных (без downsampling) данных
thresholds = np.linspace(0.001, 0.3, 100)
precisions, recalls = [], []
for t in thresholds:
    pred_pos = true_p >= t
    tp = np.sum(pred_pos & (clicks == 1))
    fp = np.sum(pred_pos & (clicks == 0))
    fn = np.sum(~pred_pos & (clicks == 1))
    precisions.append(tp / (tp + fp) if (tp + fp) > 0 else np.nan)
    recalls.append(tp / (tp + fn) if (tp + fn) > 0 else np.nan)

fig2, ax2 = plt.subplots(figsize=(7, 4.5))
ax2.plot(recalls, precisions, color="steelblue")
idx = np.argmin(np.abs(thresholds - threshold))
ax2.scatter([recalls[idx]], [precisions[idx]], color="red", zorder=5, label=f"порог = {threshold:.3f}")
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall curve")
ax2.legend()
st.pyplot(fig2)

col1, col2 = st.columns(2)
col1.metric("Precision при выбранном пороге", f"{precisions[idx]:.3f}" if not np.isnan(precisions[idx]) else "—")
col2.metric("Recall при выбранном пороге", f"{recalls[idx]:.3f}" if not np.isnan(recalls[idx]) else "—")
