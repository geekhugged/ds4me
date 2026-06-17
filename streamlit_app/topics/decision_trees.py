import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import make_moons, make_classification

st.title("10. Деревья решений")
st.caption("Теория и формулы — pages/ml-theory.html, раздел 10.")

st.markdown(
    r"""
Индекс Джини в узле: $\text{Gini}(t) = 1 - \sum_{k=1}^{K} p_k^2$.

Прирост информации при разбиении: $\text{Gain}(t,\text{split}) = I(t) - \left(\dfrac{N_L}{N_t} I(t_L) + \dfrac{N_R}{N_t} I(t_R)\right)$,
где $I(\cdot)$ — Gini или Entropy.
"""
)

st.header("Границы решений и структура дерева на синтетических данных")

dataset_type = st.radio("Датасет", ["Полумесяцы (moons)", "Линейно разделимый"], horizontal=True)
max_depth = st.slider("Максимальная глубина дерева", 1, 12, 3, 1)
min_samples_leaf = st.slider("Минимум объектов в листе", 1, 30, 1, 1)
noise = st.slider("Шум данных", 0.0, 0.5, 0.25, 0.01)
criterion = st.radio("Критерий разбиения", ["gini", "entropy"], horizontal=True)

rng_seed = 0
if dataset_type == "Полумесяцы (moons)":
    X, y = make_moons(n_samples=300, noise=noise, random_state=rng_seed)
else:
    X, y = make_classification(
        n_samples=300, n_features=2, n_redundant=0, n_informative=2,
        n_clusters_per_class=1, class_sep=2.0 - 3 * noise, random_state=rng_seed,
    )

clf = DecisionTreeClassifier(
    max_depth=max_depth, min_samples_leaf=min_samples_leaf, criterion=criterion, random_state=0
)
clf.fit(X, y)

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
Z = clf.predict(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolor="k", s=25)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title(f"Границы решений (глубина={max_depth}, лист≥{min_samples_leaf})")
st.pyplot(fig)

train_acc = clf.score(X, y)
st.write(
    f"Точность на обучающих данных: **{train_acc:.2%}**. Число листьев: **{clf.get_n_leaves()}**, "
    f"фактическая глубина: **{clf.get_depth()}**."
)

st.header("Структура дерева")

fig2, ax2 = plt.subplots(figsize=(14, 7))
plot_tree(
    clf, filled=True, feature_names=["x1", "x2"], class_names=["0", "1"],
    ax=ax2, fontsize=8, max_depth=min(max_depth, 4),
)
st.pyplot(fig2)
if max_depth > 4:
    st.caption("Для читаемости визуализация структуры дерева ограничена глубиной 4.")

st.markdown(
    """
Чем больше глубина дерева и меньше минимальное число объектов в листе, тем сложнее граница
решения и тем выше риск переобучения (дерево начинает подстраиваться под шум и отдельные точки).
Ограничение глубины и размера листа — это **pre-pruning**, базовый способ регуляризации дерева.
"""
)
