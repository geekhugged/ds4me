import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler

st.title("15. Кластеризация: k-means, иерархическая кластеризация, DBSCAN")
st.caption("Теория и формулы — pages/ml-theory.html, раздел 15.")

st.markdown(
    r"""
k-means минимизирует внутрикластерную сумму квадратов:
$J = \sum_{j=1}^k \sum_{x_i \in C_j} \|x_i - \mu_j\|^2$.

DBSCAN определяет кластеры через плотность: точка $p$ — корневая (core point), если
$|\{q : \|p-q\|\leq\varepsilon\}| \geq \text{minPts}$.
"""
)

st.header("Интерактивная кластеризация синтетических данных")

dataset_type = st.radio("Датасет", ["Блобы (выпуклые кластеры)", "Полумесяцы (нелинейная форма)"], horizontal=True)
algorithm = st.radio("Алгоритм", ["k-means", "Иерархическая (agglomerative)", "DBSCAN"], horizontal=True)
noise = st.slider("Шум / разброс данных", 0.05, 1.5, 0.6, 0.05)

if dataset_type == "Блобы (выпуклые кластеры)":
    n_clusters_true = st.slider("Истинное число кластеров в данных", 2, 6, 3, 1)
    X, y_true = make_blobs(n_samples=300, centers=n_clusters_true, cluster_std=noise, random_state=0)
else:
    X, y_true = make_moons(n_samples=300, noise=noise * 0.2, random_state=0)

X = StandardScaler().fit_transform(X)

labels = None
if algorithm == "k-means":
    k = st.slider("Число кластеров k", 2, 8, 3, 1)
    model = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = model.fit_predict(X)
    centers = model.cluster_centers_
elif algorithm == "Иерархическая (agglomerative)":
    k = st.slider("Число кластеров (срез дендрограммы)", 2, 8, 3, 1)
    linkage = st.radio("Linkage", ["ward", "complete", "average", "single"], horizontal=True)
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    labels = model.fit_predict(X)
    centers = None
else:
    eps = st.slider("eps (радиус окрестности)", 0.05, 1.5, 0.3, 0.05)
    min_samples = st.slider("minPts (минимум точек для core point)", 2, 20, 5, 1)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    centers = None

fig, ax = plt.subplots(figsize=(7, 6))
unique_labels = np.unique(labels)
cmap = plt.get_cmap("tab10")
for lbl in unique_labels:
    mask = labels == lbl
    if lbl == -1:
        ax.scatter(X[mask, 0], X[mask, 1], c="black", marker="x", s=30, label="шум (outliers)")
    else:
        ax.scatter(X[mask, 0], X[mask, 1], color=cmap(lbl % 10), s=30, label=f"кластер {lbl}", edgecolor="k", linewidths=0.3)

if centers is not None:
    ax.scatter(centers[:, 0], centers[:, 1], c="red", marker="*", s=250, edgecolor="black", label="центроиды", zorder=5)

ax.set_xlabel("x1 (стандартизован)")
ax.set_ylabel("x2 (стандартизован)")
ax.set_title(f"{algorithm} на датасете «{dataset_type}»")
ax.legend(loc="best", fontsize=8)
st.pyplot(fig)

n_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
n_noise = int(np.sum(labels == -1)) if -1 in unique_labels else 0
st.write(f"Найдено кластеров: **{n_found}**. Точек, помеченных как шум: **{n_noise}**.")

st.markdown(
    """
k-means и иерархическая кластеризация с заданным числом кластеров всегда разобьют все
точки на ровно k групп — даже если реальная структура данных этому не соответствует
(например, на «полумесяцах» k-means обычно проводит границу прямой линией через
середину, не улавливая истинную нелинейную форму кластеров). DBSCAN, напротив, способен
находить кластеры произвольной формы и явно выделяет шум, но требует подбора eps и
minPts — слишком маленький eps превратит весь датасет в шум, слишком большой —
объединит все точки в один кластер.
"""
)
