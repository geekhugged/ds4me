import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("2. Линейная регрессия и метод наименьших квадратов")
st.caption("Теория и формулы — pages/ml-theory.html, раздел 2.")

rng = np.random.default_rng(1)
n_points = 40
true_theta0, true_theta1 = 2.0, 1.5
X = np.sort(rng.uniform(0, 10, size=n_points))
y = true_theta0 + true_theta1 * X + rng.normal(scale=1.5, size=n_points)


def cost(theta0, theta1):
    pred = theta0 + theta1 * X
    return np.mean((pred - y) ** 2) / 2


st.header("Вручную подбираем параметры и смотрим на функцию стоимости")

st.markdown(
    r"""
Модель: $h_\theta(x) = \theta_0 + \theta_1 x$. Функция стоимости (MSE/2):
$J(\theta_0, \theta_1) = \frac{1}{2m}\sum_{i=1}^{m}\left(h_\theta(x^{(i)}) - y^{(i)}\right)^2$.

Подвигайте $\theta_0$ и $\theta_1$ и посмотрите, как меняется прямая и значение $J(\theta)$.
"""
)

col1, col2 = st.columns(2)
with col1:
    theta0 = st.slider("θ₀ (свободный член)", -5.0, 10.0, 0.0, 0.1)
with col2:
    theta1 = st.slider("θ₁ (наклон)", -2.0, 4.0, 0.0, 0.1)

j_value = cost(theta0, theta1)
j_best = cost(true_theta0, true_theta1)

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(X, y, label="данные")
x_line = np.array([X.min(), X.max()])
ax.plot(x_line, theta0 + theta1 * x_line, color="red", label=f"h(x) = {theta0:.2f} + {theta1:.2f}x")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"J(θ) = {j_value:.3f}")
ax.legend()
st.pyplot(fig)

st.write(
    f"Минимально достижимая стоимость (на истинных параметрах генерации данных "
    f"θ₀={true_theta0}, θ₁={true_theta1}): J ≈ **{j_best:.3f}**. "
    f"Текущая стоимость: **{j_value:.3f}**."
)

st.header("Градиентный спуск vs нормальное уравнение")

st.markdown(
    r"""
**Градиентный спуск** обновляет параметры по правилу
$\theta_j \leftarrow \theta_j - \alpha \frac{\partial J}{\partial \theta_j}$,
постепенно приближаясь к минимуму $J(\theta)$.

**Нормальное уравнение** находит точное решение за один шаг:
$\theta = (X^T X)^{-1} X^T y$.
"""
)

alpha = st.slider("Скорость обучения α", 0.001, 0.05, 0.01, 0.001)
n_iter = st.slider("Число итераций градиентного спуска", 1, 200, 50, 1)

if st.button("Запустить градиентный спуск"):
    X_design = np.column_stack([np.ones_like(X), X])

    theta_gd = np.zeros(2)
    history = []
    for _ in range(n_iter):
        pred = X_design @ theta_gd
        grad = X_design.T @ (pred - y) / n_points
        theta_gd = theta_gd - alpha * grad
        history.append(cost(theta_gd[0], theta_gd[1]))

    theta_normal, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)

    fig2, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history)
    axes[0].set_title("J(θ) на каждой итерации градиентного спуска")
    axes[0].set_xlabel("итерация")
    axes[0].set_ylabel("J(θ)")

    axes[1].scatter(X, y, label="данные")
    axes[1].plot(x_line, theta_gd[0] + theta_gd[1] * x_line, color="red",
                  label=f"GD: {theta_gd[0]:.2f} + {theta_gd[1]:.2f}x")
    axes[1].plot(x_line, theta_normal[0] + theta_normal[1] * x_line, color="green", linestyle="--",
                  label=f"Норм. ур-е: {theta_normal[0]:.2f} + {theta_normal[1]:.2f}x")
    axes[1].set_title("Сравнение решений")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].legend()

    st.pyplot(fig2)

    st.write(
        f"После {n_iter} итераций градиентный спуск даёт "
        f"θ₀ = **{theta_gd[0]:.3f}**, θ₁ = **{theta_gd[1]:.3f}** "
        f"(J = {history[-1]:.3f}); нормальное уравнение даёт точное решение "
        f"θ₀ = **{theta_normal[0]:.3f}**, θ₁ = **{theta_normal[1]:.3f}** "
        f"(J = {cost(*theta_normal):.3f}). "
        f"При большем числе итераций или α решения сближаются."
    )
