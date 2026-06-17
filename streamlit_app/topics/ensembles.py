import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_moons

st.title("11. Ансамбли: Bagging, Random Forest, Boosting")
st.caption("Теория и формулы — pages/ml-theory.html, раздел 11.")

st.markdown(
    r"""
Bagging усредняет независимые модели, обученные на бутстрэп-подвыборках:
$\hat{f}_{\text{bag}}(x) = \frac{1}{B}\sum_{b=1}^{B}\hat{f}_b(x)$ — это снижает variance.

Gradient Boosting строит ансамбль последовательно, на каждом шаге приближая
антиградиент функции потерь: $F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$, где $\eta$ — learning rate.
"""
)

st.header("Сравнение границ решений: одно дерево vs Random Forest vs Gradient Boosting")

noise = st.slider("Шум данных", 0.05, 0.5, 0.25, 0.01)
n_estimators = st.slider("Число деревьев в ансамбле (n_estimators)", 1, 200, 50, 1)
max_depth_single = st.slider("Глубина одиночного дерева", 1, 10, 4, 1)
learning_rate = st.slider("Learning rate (для Gradient Boosting)", 0.01, 1.0, 0.1, 0.01)

X, y = make_moons(n_samples=300, noise=noise, random_state=0)

single_tree = DecisionTreeClassifier(max_depth=max_depth_single, random_state=0)
rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth_single, random_state=0)
gb = GradientBoostingClassifier(
    n_estimators=n_estimators, learning_rate=learning_rate, max_depth=2, random_state=0
)

models = {
    "Одно дерево решений": single_tree,
    "Random Forest (bagging)": rf,
    "Gradient Boosting": gb,
}

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, model) in zip(axes, models.items()):
    model.fit(X, y)
    Z = model.predict(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolor="k", s=20)
    acc = model.score(X, y)
    ax.set_title(f"{name}\ntrain acc={acc:.2%}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

st.pyplot(fig)

st.markdown(
    """
Одиночное дерево с достаточной глубиной может почти идеально подстроиться под шум
обучающих данных (переобучение, высокая variance). Random Forest усредняет много
независимых деревьев, обученных на случайных подвыборках и подмножествах признаков,
сглаживая границу решения. Gradient Boosting строит ансамбль последовательно,
постепенно снижая bias за счёт исправления ошибок предыдущих моделей — при большом
числе итераций и высоком learning rate он тоже может переобучиться.
"""
)

st.header("Влияние числа базовых моделей на качество (bagging vs boosting)")

n_range = np.unique(np.clip(np.linspace(1, max(n_estimators, 5), 12).astype(int), 1, None))
rf_scores = []
gb_scores = []
for n in n_range:
    rf_n = RandomForestClassifier(n_estimators=int(n), max_depth=max_depth_single, random_state=0)
    gb_n = GradientBoostingClassifier(
        n_estimators=int(n), learning_rate=learning_rate, max_depth=2, random_state=0
    )
    rf_n.fit(X, y)
    gb_n.fit(X, y)
    rf_scores.append(rf_n.score(X, y))
    gb_scores.append(gb_n.score(X, y))

fig2, ax2 = plt.subplots(figsize=(7, 4.5))
ax2.plot(n_range, rf_scores, marker="o", label="Random Forest")
ax2.plot(n_range, gb_scores, marker="s", label="Gradient Boosting")
ax2.set_xlabel("Число базовых моделей")
ax2.set_ylabel("Точность на обучающих данных")
ax2.set_title("Качество в зависимости от размера ансамбля")
ax2.legend()
st.pyplot(fig2)

st.markdown(
    """
Random Forest (bagging) обычно быстро выходит на плато: добавление новых деревьев
снижает variance, но почти не увеличивает риск переобучения. Gradient Boosting может
продолжать улучшать качество на обучающей выборке почти до 100% при большом числе
итераций — на тестовых данных это часто означает переобучение, если не контролировать
learning rate, глубину деревьев и early stopping.
"""
)
