import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix

st.title("6. Метрики качества для классификации")
st.caption("Теория и формулы — pages/ml-theory.html, раздел 6.")

st.markdown(
    r"""
Из матрицы ошибок ($TP$, $TN$, $FP$, $FN$) строятся:

$\text{Precision} = \dfrac{TP}{TP+FP}$, $\quad \text{Recall} = \dfrac{TP}{TP+FN}$,
$\quad F_1 = 2\cdot\dfrac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}$.

ROC-кривая строится в координатах $(\text{FPR}, \text{TPR})$ при варьировании порога,
PR-кривая — в координатах $(\text{Recall}, \text{Precision})$.
"""
)

st.header("Confusion matrix, ROC и PR кривые с порогом")

separation = st.slider("Разделимость классов", 0.2, 4.0, 1.5, 0.1)
n_per_class = st.slider("Объектов на класс", 20, 300, 100, 10)
threshold = st.slider("Порог классификации", 0.01, 0.99, 0.50, 0.01)

rng = np.random.default_rng(7)
# Скоры моделируем напрямую как вероятности из двух смещённых бета/нормальных распределений
neg_scores = np.clip(rng.normal(loc=0.5 - separation * 0.1, scale=0.2, size=n_per_class), 0.001, 0.999)
pos_scores = np.clip(rng.normal(loc=0.5 + separation * 0.1, scale=0.2, size=n_per_class), 0.001, 0.999)

y_true = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])
y_score = np.concatenate([neg_scores, pos_scores])

y_pred = (y_score >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
accuracy = (tp + tn) / (tp + tn + fp + fn)
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

cm = np.array([[tn, fp], [fn, tp]])
axes[0].imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16)
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(["pred 0", "pred 1"])
axes[0].set_yticklabels(["true 0", "true 1"])
axes[0].set_title("Confusion matrix")

fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
cur_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
axes[1].plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
axes[1].plot([0, 1], [0, 1], "--", color="gray", label="случайный классификатор")
axes[1].scatter([cur_fpr], [recall], color="red", zorder=3, label=f"порог={threshold:.2f}")
axes[1].set_xlabel("FPR")
axes[1].set_ylabel("TPR (recall)")
axes[1].set_title("ROC-кривая")
axes[1].legend(fontsize=8)

prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_score)
pr_auc = auc(rec_curve, prec_curve)
axes[2].plot(rec_curve, prec_curve, label=f"PR (AUC={pr_auc:.3f})")
axes[2].scatter([recall], [precision], color="red", zorder=3, label=f"порог={threshold:.2f}")
axes[2].set_xlabel("Recall")
axes[2].set_ylabel("Precision")
axes[2].set_title("PR-кривая")
axes[2].legend(fontsize=8)

st.pyplot(fig)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy:.3f}")
col2.metric("Precision", f"{precision:.3f}")
col3.metric("Recall", f"{recall:.3f}")
col4.metric("F1", f"{f1:.3f}")

st.markdown(
    """
Подвигайте порог классификации: при росте порога модель становится «осторожнее» —
обычно растёт precision, но падает recall. ROC-AUC и PR-AUC не зависят от выбора
конкретного порога и агрегируют качество по всем порогам сразу.
"""
)
