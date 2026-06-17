import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

st.title("9. Кросс-валидация и подбор гиперпараметров")
st.caption("Теория и формулы — pages/ml-theory.html, раздел 9.")

st.markdown(
    r"""
K-Fold кросс-валидация: данные делятся на $k$ фолдов, модель обучается $k$ раз на $D \setminus F_i$
и проверяется на $F_i$. Итоговая оценка — среднее метрики по фолдам:
$\overline{s} = \dfrac{1}{k}\sum_{i=1}^{k} s_i$, со стандартной ошибкой
$\text{SE} = \dfrac{1}{\sqrt{k}}\sqrt{\dfrac{1}{k-1}\sum_i (s_i-\overline{s})^2}$.
"""
)

st.header("Визуализация фолдов")

n_samples = st.slider("Число объектов", 10, 100, 30, 1)
k_folds = st.slider("Число фолдов k", 2, 10, 5, 1)

indices = np.arange(n_samples)
kf = KFold(n_splits=k_folds, shuffle=True, random_state=0)

fig, ax = plt.subplots(figsize=(9, 0.6 * k_folds + 1))
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
    row = np.zeros(n_samples)
    row[val_idx] = 1
    ax.scatter(indices, [fold_idx] * n_samples, c=row, cmap="coolwarm", s=25, vmin=0, vmax=1)
ax.set_yticks(range(k_folds))
ax.set_yticklabels([f"фолд {i+1}" for i in range(k_folds)])
ax.set_xlabel("индекс объекта")
ax.set_title("Синий = train, красный = validation в каждом фолде")
st.pyplot(fig)

st.header("Подбор гиперпараметра λ (Ridge) через кросс-валидацию")

n_data = st.slider("Число точек данных", 20, 200, 50, 5)
noise = st.slider("Шум", 0.0, 2.0, 0.5, 0.1)
degree = st.slider("Степень полиномиальных признаков", 1, 10, 5, 1)
k_cv = st.slider("Число фолдов для подбора λ", 3, 10, 5, 1)

rng = np.random.default_rng(1)
X = rng.uniform(-1, 1, size=n_data)
y = np.sin(2 * np.pi * X) + rng.normal(scale=noise, size=n_data)

lambdas = np.logspace(-4, 2, 25)
means, stds = [], []
for lam in lambdas:
    model = make_pipeline(PolynomialFeatures(degree=degree), Ridge(alpha=lam))
    scores = cross_val_score(
        model, X.reshape(-1, 1), y, cv=KFold(n_splits=k_cv, shuffle=True, random_state=0),
        scoring="neg_mean_squared_error",
    )
    means.append(-scores.mean())
    stds.append(scores.std())

means = np.array(means)
stds = np.array(stds)
best_idx = np.argmin(means)

fig2, ax2 = plt.subplots(figsize=(8, 4.5))
ax2.plot(lambdas, means, marker="o", markersize=3, label="средняя MSE по фолдам")
ax2.fill_between(lambdas, means - stds, means + stds, alpha=0.2, label="± std по фолдам")
ax2.axvline(lambdas[best_idx], color="red", linestyle="--", label=f"лучшая λ={lambdas[best_idx]:.4f}")
ax2.set_xscale("log")
ax2.set_xlabel("λ (лог. шкала)")
ax2.set_ylabel("CV MSE")
ax2.set_title("Кривая валидации: MSE и разброс между фолдами")
ax2.legend()
st.pyplot(fig2)

st.markdown(
    f"""
Лучшее найденное значение λ = **{lambdas[best_idx]:.4f}**, средняя CV MSE = **{means[best_idx]:.4f}**
± **{stds[best_idx]:.4f}**.

Чем шире полоса разброса (± std), тем менее надёжна оценка метрики на конкретном значении
гиперпараметра — стоит ориентироваться не только на среднее, но и на стабильность по фолдам.
"""
)
