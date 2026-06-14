import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LogisticRegression

st.title("5. Логистическая регрессия и функция потерь log-loss")
st.caption("Теория и формулы — pages/ml-theory.html, раздел 5.")

st.markdown(
    r"""
Логистическая регрессия моделирует вероятность класса 1 через сигмоиду:
$h_\theta(x) = \sigma(\theta^T x) = \dfrac{1}{1 + e^{-\theta^T x}}$.

Функция потерь (log-loss) для одного примера:
$L(y, \hat{y}) = -\left[y \log \hat{y} + (1-y)\log(1-\hat{y})\right]$.
"""
)

st.header("Решающая граница и сигмоида")

separation = st.slider("Разделимость классов (расстояние между центрами)", 0.5, 5.0, 2.0, 0.1)
n_per_class = st.slider("Точек на класс", 10, 150, 50, 10)

rng = np.random.default_rng(5)
X0 = rng.normal(loc=[-separation / 2, 0], scale=1.0, size=(n_per_class, 2))
X1 = rng.normal(loc=[separation / 2, 0], scale=1.0, size=(n_per_class, 2))
X = np.vstack([X0, X1])
y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])

clf = LogisticRegression().fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
Z = clf.predict_proba(np.column_stack([xx.ravel(), yy.ravel()]))[:, 1].reshape(xx.shape)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].contourf(xx, yy, Z, levels=20, cmap="RdBu_r", alpha=0.6)
axes[0].contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2)
axes[0].scatter(X0[:, 0], X0[:, 1], label="класс 0", edgecolor="k")
axes[0].scatter(X1[:, 0], X1[:, 1], label="класс 1", edgecolor="k")
axes[0].set_title("Решающая граница (p = 0.5)")
axes[0].set_xlabel("x1")
axes[0].set_ylabel("x2")
axes[0].legend()

z = np.linspace(-10, 10, 200)
sigma = 1 / (1 + np.exp(-z))
axes[1].plot(z, sigma)
axes[1].axhline(0.5, color="gray", linestyle="--")
axes[1].axvline(0, color="gray", linestyle="--")
axes[1].set_title(r"Сигмоида $\sigma(z) = 1/(1+e^{-z})$")
axes[1].set_xlabel("z")
axes[1].set_ylabel(r"$\sigma(z)$")

st.pyplot(fig)

acc = clf.score(X, y)
st.write(f"Точность модели на обучающих данных: **{acc:.2%}**.")

st.header("Калькулятор log-loss")

st.markdown(
    "Выберите истинный класс $y$ и предсказанную моделью вероятность $\\hat{y}$, "
    "чтобы увидеть, как сильно штрафуется ошибка."
)

col1, col2 = st.columns(2)
with col1:
    y_true = st.radio("Истинный класс y", [0, 1], horizontal=True)
with col2:
    p_pred = st.slider("Предсказанная вероятность класса 1 (ŷ)", 0.001, 0.999, 0.5, 0.001)

loss = -(y_true * np.log(p_pred) + (1 - y_true) * np.log(1 - p_pred))

p_range = np.linspace(0.001, 0.999, 200)
if y_true == 1:
    loss_curve = -np.log(p_range)
else:
    loss_curve = -np.log(1 - p_range)

fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.plot(p_range, loss_curve, label=f"L(y={y_true}, ŷ)")
ax2.scatter([p_pred], [loss], color="red", zorder=3, label=f"ŷ = {p_pred:.3f}, L = {loss:.3f}")
ax2.set_xlabel(r"$\hat{y}$ — предсказанная вероятность класса 1")
ax2.set_ylabel("log-loss")
ax2.legend()
st.pyplot(fig2)

st.write(
    f"При y = **{y_true}** и ŷ = **{p_pred:.3f}** значение log-loss = **{loss:.3f}**. "
    "Чем увереннее модель ошибается (предсказывает вероятность, далёкую от истинного класса), "
    "тем сильнее (нелинейно) растёт штраф."
)
