import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import make_blobs, make_swiss_roll, make_s_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

st.title("17. Снижение размерности: PCA, SVD, t-SNE, UMAP")
st.caption("Теория и формулы — pages/ml-theory.html, раздел 17.")

st.markdown(
    r"""
PCA ищет ортогональные направления максимальной дисперсии. Главные компоненты — это
собственные векторы ковариационной матрицы $C = \tfrac{1}{n} X^\top X$ (для центрированных
данных), а дисперсия вдоль $i$-й компоненты равна собственному значению $\lambda_i$.
Доля объяснённой дисперсии:

$$ \text{EVR}_i = \frac{\lambda_i}{\sum_{j} \lambda_j}. $$

t-SNE — нелинейный метод: он сохраняет локальные соседства, моделируя сходства точек
вероятностями $p_{ij}$ в исходном пространстве и $q_{ij}$ в проекции, и минимизирует
дивергенцию Кульбака — Лейблера $\mathrm{KL}(P\,\|\,Q) = \sum_{i\ne j} p_{ij}\log\frac{p_{ij}}{q_{ij}}$.
Параметр *perplexity* задаёт эффективное число соседей. UMAP здесь не используется
(библиотека недоступна), но идейно близок t-SNE и обычно быстрее.
"""
)

st.header("Интерактивная проекция в 2D")

dataset_type = st.radio(
    "Датасет",
    ["Блобы (несколько кластеров)", "Swiss roll (рулет)", "S-curve (S-образное многообразие)"],
    horizontal=True,
)
method = st.radio("Метод снижения размерности", ["PCA", "t-SNE"], horizontal=True)
noise = st.slider("Уровень шума", 0.0, 2.0, 0.3, 0.05)
n_samples = st.slider("Число точек", 100, 500, 300, 50)


@st.cache_data
def make_dataset(dataset_type, noise, n_samples):
    if dataset_type == "Блобы (несколько кластеров)":
        X, color = make_blobs(
            n_samples=n_samples, centers=4, n_features=3, cluster_std=0.8 + noise, random_state=0
        )
        color = color.astype(float)
    elif dataset_type == "Swiss roll (рулет)":
        X, color = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=0)
    else:
        X, color = make_s_curve(n_samples=n_samples, noise=noise * 0.1, random_state=0)
    return X, color


X, color = make_dataset(dataset_type, noise, n_samples)
X = StandardScaler().fit_transform(X)

if method == "PCA":
    n_components = st.slider("Число компонент (для scree plot)", 2, X.shape[1], min(3, X.shape[1]), 1)
    pca_full = PCA(n_components=n_components).fit(X)
    X_2d = pca_full.transform(X)[:, :2]
else:
    perplexity = st.slider("Perplexity (число соседей)", 5, 50, 30, 1)
    perplexity = min(perplexity, (n_samples - 1) // 3)


@st.cache_data
def run_tsne(X, perplexity, n_samples, dataset_type, noise):
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=0)
    return tsne.fit_transform(X)


if method == "t-SNE":
    X_2d = run_tsne(X, perplexity, n_samples, dataset_type, noise)

if method == "PCA":
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sc = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=color, cmap="viridis", s=18, edgecolor="k", linewidths=0.2)
    axes[0].set_xlabel("PC 1")
    axes[0].set_ylabel("PC 2")
    axes[0].set_title("Проекция на 2 главные компоненты")
    fig.colorbar(sc, ax=axes[0], shrink=0.8)

    evr = pca_full.explained_variance_ratio_
    comp_idx = np.arange(1, len(evr) + 1)
    axes[1].bar(comp_idx, evr, alpha=0.7, label="доля дисперсии")
    axes[1].plot(comp_idx, np.cumsum(evr), "-o", color="red", label="накопленная доля")
    axes[1].set_xlabel("номер компоненты")
    axes[1].set_ylabel("объяснённая дисперсия")
    axes[1].set_title("Scree plot")
    axes[1].set_xticks(comp_idx)
    axes[1].legend(fontsize=8)
    st.pyplot(fig)

    st.write(
        f"Первые 2 компоненты объясняют **{evr[:2].sum() * 100:.1f}%** дисперсии, "
        f"все {n_components} компонент(ы) — **{evr.sum() * 100:.1f}%**."
    )
else:
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=color, cmap="viridis", s=18, edgecolor="k", linewidths=0.2)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"t-SNE (perplexity={perplexity})")
    fig.colorbar(sc, ax=ax, shrink=0.8)
    st.pyplot(fig)

st.markdown(
    """
PCA — линейный метод: он хорошо «разворачивает» данные, лежащие близко к линейному
подпространству, и даёт интерпретируемые оси с долей объяснённой дисперсии. Но на
сильно изогнутых многообразиях (swiss roll, S-curve) PCA «склеивает» далёкие по
геодезическому расстоянию точки.

t-SNE сохраняет локальную структуру и часто красиво разделяет кластеры, однако:
расстояния между кластерами и их размеры на t-SNE-карте не интерпретируемы; результат
зависит от perplexity и случайной инициализации; метод вычислительно тяжёлый. Используйте
t-SNE/UMAP для визуализации, а PCA — ещё и для предобработки и сжатия признаков.
"""
)
