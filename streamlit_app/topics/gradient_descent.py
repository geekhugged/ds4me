import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("7. Градиентный спуск и его вариации")
st.caption("Теория и формулы — pages/ml-theory.html, раздел 7.")

st.markdown(
    r"""
Базовое правило обновления параметров: $\theta_{t+1} = \theta_t - \alpha \nabla_\theta J(\theta_t)$.

Momentum добавляет инерцию: $v_t = \beta v_{t-1} + (1-\beta)\nabla_\theta J(\theta_t)$,
$\theta_{t+1} = \theta_t - \alpha v_t$.

Ниже — траектории спуска на «овражной» квадратичной поверхности потерь
$J(x, y) = a x^2 + b y^2$ для разных оптимизаторов.
"""
)

st.header("Траектории на 2D loss surface")

a_coef = st.slider("Кривизна по x (a)", 0.1, 10.0, 1.0, 0.1)
b_coef = st.slider("Кривизна по y (b, овражность)", 0.1, 10.0, 8.0, 0.1)
lr = st.slider("Learning rate (α)", 0.01, 1.0, 0.15, 0.01)
momentum_beta = st.slider("Momentum β", 0.0, 0.99, 0.9, 0.01)
n_steps = st.slider("Число шагов", 5, 100, 30, 1)
optimizer = st.radio("Оптимизатор", ["Batch GD (без momentum)", "Momentum", "Adam"], horizontal=True)

start = np.array([4.0, 4.0])


def loss(p):
    return a_coef * p[0] ** 2 + b_coef * p[1] ** 2


def grad(p):
    return np.array([2 * a_coef * p[0], 2 * b_coef * p[1]])


path = [start.copy()]
theta = start.copy()

if optimizer == "Batch GD (без momentum)":
    for _ in range(n_steps):
        theta = theta - lr * grad(theta)
        path.append(theta.copy())

elif optimizer == "Momentum":
    v = np.zeros(2)
    for _ in range(n_steps):
        g = grad(theta)
        v = momentum_beta * v + (1 - momentum_beta) * g
        theta = theta - lr * v
        path.append(theta.copy())

else:  # Adam
    m = np.zeros(2)
    v = np.zeros(2)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for t in range(1, n_steps + 1):
        g = grad(theta)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
        path.append(theta.copy())

path = np.array(path)

xx, yy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
zz = a_coef * xx ** 2 + b_coef * yy ** 2

fig, ax = plt.subplots(figsize=(7, 6))
ax.contour(xx, yy, zz, levels=25, cmap="viridis")
ax.plot(path[:, 0], path[:, 1], "o-", color="red", markersize=3, label="траектория")
ax.scatter([0], [0], color="black", marker="*", s=150, zorder=5, label="минимум")
ax.scatter([start[0]], [start[1]], color="blue", marker="s", s=80, zorder=5, label="старт")
ax.set_xlabel("θ₁")
ax.set_ylabel("θ₂")
ax.set_title(f"{optimizer}: {n_steps} шагов")
ax.legend()
st.pyplot(fig)

losses = [loss(p) for p in path]
fig2, ax2 = plt.subplots(figsize=(7, 3.5))
ax2.plot(losses, marker="o", markersize=3)
ax2.set_yscale("log")
ax2.set_xlabel("шаг")
ax2.set_ylabel("J(θ) (лог. шкала)")
ax2.set_title("Сходимость функции потерь")
st.pyplot(fig2)

st.markdown(
    f"""
Финальная точка: θ = ({path[-1, 0]:.3f}, {path[-1, 1]:.3f}), J(θ) = {losses[-1]:.5f}.

При сильной «овражности» (b ≫ a) обычный батч-градиентный спуск зигзагит и сходится медленно.
Momentum гасит колебания вдоль «крутого» направления и ускоряет движение вдоль «плоского».
Adam адаптирует шаг отдельно для каждой координаты, что часто даёт более устойчивую сходимость.
"""
)
