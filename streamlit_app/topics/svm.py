import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_blobs
from sklearn.preprocessing import StandardScaler

st.title("12. Метод опорных векторов (SVM) и kernel trick")
st.caption("Теория и формулы — pages/ml-theory.html, раздел 12.")

st.markdown(
    r"""
Soft margin SVM решает задачу оптимизации:

$$ \min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^m \xi_i \quad \text{при } y_i(w^Tx_i+b) \geq 1-\xi_i,\;\xi_i\geq 0 $$

Kernel trick заменяет скалярное произведение $\phi(x_i)^T\phi(x_j)$ функцией ядра $K(x_i,x_j)$, например:

$$ K_{\text{RBF}}(x_i,x_j) = \exp(-\gamma\|x_i-x_j\|^2) $$
"""
)

st.header("Границы решений SVM с разными ядрами, C и gamma")

dataset_type = st.radio("Датасет", ["Полумесяцы (moons)", "Два блоба (linearly separable)"], horizontal=True)
kernel = st.radio("Ядро", ["linear", "poly", "rbf"], horizontal=True)
C = st.slider("C (силa штрафа за нарушения зазора)", 0.01, 100.0, 1.0, 0.01)
gamma = st.slider("gamma (только для rbf/poly)", 0.01, 10.0, 1.0, 0.01)
noise = st.slider("Шум данных", 0.05, 0.5, 0.2, 0.01)
degree = 3
if kernel == "poly":
    degree = st.slider("Степень полиномиального ядра", 2, 6, 3, 1)

if dataset_type == "Полумесяцы (moons)":
    X, y = make_moons(n_samples=200, noise=noise, random_state=0)
else:
    X, y = make_blobs(n_samples=200, centers=2, cluster_std=1.0 + noise * 3, random_state=0)

scaler = StandardScaler()
X = scaler.fit_transform(X)

clf = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
clf.fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
Z = clf.decision_function(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(7, 6))
ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 30), cmap="RdBu", alpha=0.5)
ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=["gray", "black", "gray"], linestyles=["--", "-", "--"])
ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolor="k", s=25, zorder=3)
ax.scatter(
    clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
    s=150, facecolors="none", edgecolors="lime", linewidths=2, label="Опорные векторы", zorder=4,
)
ax.set_xlabel("x1 (стандартизован)")
ax.set_ylabel("x2 (стандартизован)")
ax.set_title(f"SVM, kernel={kernel}, C={C:.2f}, gamma={gamma:.2f}")
ax.legend(loc="upper right", fontsize=8)
st.pyplot(fig)

train_acc = clf.score(X, y)
st.write(
    f"Точность на обучающих данных: **{train_acc:.2%}**. "
    f"Число опорных векторов: **{len(clf.support_)}** из {len(X)} объектов."
)

st.markdown(
    """
Чёрная линия — разделяющая гиперплоскость (граница решения), серые штриховые линии —
границы зазора ($w^Tx+b = \\pm 1$). Точки в зелёных кружках — опорные векторы, которые
определяют положение границы. При увеличении `C` модель сильнее штрафует нарушения
зазора (граница «подгоняется» под обучающие данные, риск переобучения). При увеличении
`gamma` (для rbf) область влияния каждой точки сужается, граница становится более
изломанной и локальной.
"""
)
