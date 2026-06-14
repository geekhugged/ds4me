import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans

st.title("1. Постановка задачи ML: supervised / unsupervised / reinforcement learning")
st.caption("Теория и формулы — pages/ml-theory.html, раздел 1.")

st.header("Supervised vs unsupervised: один набор данных, разные задачи")

rng = np.random.default_rng(42)
n_per_cluster = 40
centers = np.array([[2.0, 2.0], [7.0, 7.0], [2.0, 7.0]])
X = np.vstack([c + rng.normal(scale=0.9, size=(n_per_cluster, 2)) for c in centers])
true_labels = np.repeat(np.arange(3), n_per_cluster)
y_reg = X[:, 0] * 1.5 + X[:, 1] * 0.5 + rng.normal(scale=1.0, size=X.shape[0])

mode = st.radio(
    "Как мы смотрим на эти данные?",
    [
        "Без разметки (unsupervised)",
        "С разметкой: классификация (supervised)",
        "С разметкой: регрессия (supervised)",
    ],
)

fig, ax = plt.subplots(figsize=(6, 5))

if mode == "Без разметки (unsupervised)":
    k = st.slider("Число кластеров k для k-means", 2, 5, 3)
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    ax.scatter(X[:, 0], X[:, 1], c=km.labels_, cmap="tab10")
    ax.set_title("Unsupervised: меток нет, k-means сам нашёл структуру")
elif mode == "С разметкой: классификация (supervised)":
    ax.scatter(X[:, 0], X[:, 1], c=true_labels, cmap="tab10")
    ax.set_title("Supervised (классификация): у каждой точки есть метка класса")
else:
    sc = ax.scatter(X[:, 0], X[:, 1], c=y_reg, cmap="viridis")
    fig.colorbar(sc, ax=ax, label="y (целевая переменная)")
    ax.set_title("Supervised (регрессия): у каждой точки есть число y")

ax.set_xlabel("x1")
ax.set_ylabel("x2")
st.pyplot(fig)

st.markdown(
    """
Это **один и тот же набор точек** $X$. Разница — в том, какая
дополнительная информация (метки или целевая переменная) у нас есть и что
мы хотим получить на выходе: структуру данных (unsupervised), класс
(классификация) или число (регрессия).
"""
)

st.header("Reinforcement learning: explore vs exploit (multi-armed bandit)")

st.markdown(
    r"""
Агент на каждом шаге $t$ выбирает одно из $k$ действий $a_t$ и получает
награду $r_t$. Цель — максимизировать суммарную награду $\sum_t r_t$, как в
формуле $\pi^*$ из теории. Стратегия **$\varepsilon$-greedy**: с
вероятностью $\varepsilon$ выбираем случайное действие (exploration),
иначе — действие с наибольшей оценённой средней наградой (exploitation).
"""
)

n_arms = st.slider("Число действий (рук)", 2, 10, 5)
epsilon = st.slider("ε — вероятность случайного исследования", 0.0, 1.0, 0.1, 0.01)
n_steps = st.slider("Число шагов", 50, 2000, 500, 50)

if st.button("Запустить симуляцию бандита"):
    rng2 = np.random.default_rng(0)
    true_means = rng2.uniform(0, 1, size=n_arms)
    estimates = np.zeros(n_arms)
    counts = np.zeros(n_arms)
    rewards = np.zeros(n_steps)

    for t in range(n_steps):
        if rng2.random() < epsilon:
            a = rng2.integers(n_arms)
        else:
            a = int(np.argmax(estimates))
        r = rng2.normal(true_means[a], 0.2)
        counts[a] += 1
        estimates[a] += (r - estimates[a]) / counts[a]
        rewards[t] = r

    fig2, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(np.cumsum(rewards) / (np.arange(n_steps) + 1))
    axes[0].axhline(true_means.max(), color="green", linestyle="--", label="лучшая рука")
    axes[0].set_title("Средняя награда нарастающим итогом")
    axes[0].set_xlabel("шаг")
    axes[0].legend()

    axes[1].bar(np.arange(n_arms) - 0.2, true_means, width=0.4, label="истинная награда")
    axes[1].bar(np.arange(n_arms) + 0.2, estimates, width=0.4, label="оценка агента")
    axes[1].set_title("Истинная и оценённая награда по действиям")
    axes[1].set_xlabel("действие")
    axes[1].legend()

    st.pyplot(fig2)
    st.write(
        f"Лучшее действие по истинным значениям: **{int(np.argmax(true_means))}**, "
        f"агент выбирал чаще всего: **{int(np.argmax(counts))}**."
    )
