import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

st.title("19. Работа с несбалансированными данными")
st.caption("Теория и формулы — pages/ml-theory.html, раздел 19.")

st.markdown(
    r"""
При сильном дисбалансе accuracy обманчива: классификатор, всегда предсказывающий
мажоритарный класс, получит высокую точность, но нулевой recall по редкому классу.
Поэтому смотрят на precision/recall/F1 и PR-AUC.

Стратегии борьбы с дисбалансом:

- **class_weight='balanced'**: вес класса обратно пропорционален его частоте,
  $w_c = \dfrac{n}{K\, n_c}$, что усиливает вклад редкого класса в функцию потерь.
- **Oversampling**: дублирование (или синтез, например SMOTE) объектов редкого класса
  до баланса. Здесь SMOTE недоступен — используем простой ресэмплинг с повторением.

Подбор **порога** классификации $t$ ($\hat{y}=1$, если $p \ge t$) — отдельный рычаг:
снижение порога повышает recall ценой precision.
"""
)

st.header("Интерактивная демонстрация")

minority_frac = st.slider("Доля редкого (положительного) класса", 0.02, 0.5, 0.1, 0.01)
separation = st.slider("Разделимость классов", 0.5, 4.0, 1.5, 0.1)
strategy = st.radio(
    "Стратегия", ["Без обработки", "class_weight='balanced'", "Oversampling (resample)"], horizontal=True
)
threshold = st.slider("Порог классификации", 0.01, 0.99, 0.50, 0.01)

n_total = 1000
n_pos = max(int(minority_frac * n_total), 10)
n_neg = n_total - n_pos

rng = np.random.default_rng(3)
X_neg = rng.normal(loc=0.0, scale=1.0, size=(n_neg, 2))
X_pos = rng.normal(loc=separation, scale=1.0, size=(n_pos, 2))
X = np.vstack([X_neg, X_pos])
y = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

if strategy == "class_weight='balanced'":
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)
elif strategy == "Oversampling (resample)":
    pos_mask = y_train == 1
    X_pos_tr, y_pos_tr = X_train[pos_mask], y_train[pos_mask]
    X_neg_tr, y_neg_tr = X_train[~pos_mask], y_train[~pos_mask]
    X_pos_up, y_pos_up = resample(
        X_pos_tr, y_pos_tr, replace=True, n_samples=len(y_neg_tr), random_state=0
    )
    X_bal = np.vstack([X_neg_tr, X_pos_up])
    y_bal = np.concatenate([y_neg_tr, y_pos_up])
    model = LogisticRegression(max_iter=1000)
    model.fit(X_bal, y_bal)
else:
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

y_score = model.predict_proba(X_test)[:, 1]
y_pred = (y_score >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
accuracy = (tp + tn) / (tp + tn + fp + fn)

prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_score)
pr_auc = auc(rec_curve, prec_curve)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

cm = np.array([[tn, fp], [fn, tp]])
axes[0].imshow(cm, cmap="Oranges")
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16)
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(["pred 0", "pred 1"])
axes[0].set_yticklabels(["true 0", "true 1"])
axes[0].set_title("Confusion matrix (тест)")

baseline = y_test.mean()
axes[1].plot(rec_curve, prec_curve, label=f"PR (AUC={pr_auc:.3f})")
axes[1].axhline(baseline, ls="--", color="gray", label=f"базлайн (доля класса={baseline:.2f})")
axes[1].scatter([recall], [precision], color="red", zorder=3, label=f"порог={threshold:.2f}")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("PR-кривая")
axes[1].set_ylim(0, 1.02)
axes[1].legend(fontsize=8)
st.pyplot(fig)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy:.3f}")
col2.metric("Precision", f"{precision:.3f}")
col3.metric("Recall", f"{recall:.3f}")
col4.metric("F1", f"{f1:.3f}")

st.markdown(
    """
Поставьте долю редкого класса в 2–5% и стратегию «Без обработки»: accuracy будет высокой,
но recall по редкому классу — низким (модель почти всегда предсказывает мажоритарный класс).
Включите class_weight='balanced' или oversampling — recall вырастет ценой части precision.

Важно: дисбаланс лечат только на обучающей выборке, тест оставляют как есть, иначе метрики
будут оптимистично смещены. PR-AUC информативнее ROC-AUC при сильном дисбалансе, так как
напрямую отражает качество по редкому классу.
"""
)
