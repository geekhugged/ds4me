import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("5. Градиентный спуск и оптимизаторы (SGD, Momentum, RMSProp, Adam, AdamW)")
st.caption("Теория и формулы — pages/deep-learning.html, раздел 5.")

st.markdown(
    r"""
Обучение нейросети — задача минимизации $\mathcal{L}(\boldsymbol{\theta})$. Все оптимизаторы
строятся вокруг идеи итеративного шага в направлении антиградиента, но по-разному
адаптируют скорость обучения и сглаживают шум градиентов.

| Оптимизатор | Ключевая идея |
|---|---|
| SGD | $\theta \leftarrow \theta - \eta g$ |
| Momentum | Накопление скользящего среднего градиентов |
| RMSProp | Нормировка шага на корень из ср. кв. градиента |
| Adam | Momentum + RMSProp + коррекция смещения |
| AdamW | Adam + отдельный weight decay |
"""
)

st.header("Траектории оптимизаторов на 2D поверхности потерь")

st.markdown(
    r"""
Визуализируем траектории оптимизаторов на функции Розенброка:
$$f(x, y) = (1-x)^2 + 100(y - x^2)^2$$
Минимум: $f(1, 1) = 0$. Функция имеет узкую «банановую» долину — классическую проблему для SGD.
"""
)

col1, col2 = st.columns(2)
with col1:
    lr_sgd = st.slider("Learning rate SGD", 1e-4, 1e-2, 1e-3, 1e-4, format="%.4f")
    lr_momentum = st.slider("Learning rate Momentum", 1e-4, 1e-2, 1e-3, 1e-4, format="%.4f")
    beta_momentum = st.slider("β Momentum", 0.0, 0.99, 0.9, 0.01)
with col2:
    lr_rmsprop = st.slider("Learning rate RMSProp", 1e-4, 1e-2, 1e-3, 1e-4, format="%.4f")
    lr_adam = st.slider("Learning rate Adam", 1e-4, 1e-2, 2e-3, 1e-4, format="%.4f")
    n_steps_opt = st.slider("Число шагов оптимизации", 50, 1000, 300, 50)

x0 = st.slider("Начальная x₀", -1.5, 1.5, -1.0, 0.1)
y0 = st.slider("Начальная y₀", -1.0, 3.0, 1.5, 0.1)

def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def rosenbrock_grad(x, y):
    dx = -2 * (1 - x) - 400 * x * (y - x ** 2)
    dy = 200 * (y - x ** 2)
    g = np.array([dx, dy])
    return np.clip(g, -1e6, 1e6)

eps_opt = 1e-8

def run_sgd(x0, y0, lr, n_steps):
    path = [(x0, y0)]
    x, y = x0, y0
    for _ in range(n_steps):
        g = rosenbrock_grad(x, y)
        x -= lr * g[0]
        y -= lr * g[1]
        path.append((x, y))
    return path

def run_momentum(x0, y0, lr, beta, n_steps):
    path = [(x0, y0)]
    x, y = x0, y0
    v = np.zeros(2)
    for _ in range(n_steps):
        g = rosenbrock_grad(x, y)
        v = beta * v + (1 - beta) * g
        x -= lr * v[0]
        y -= lr * v[1]
        path.append((x, y))
    return path

def run_rmsprop(x0, y0, lr, beta2, n_steps):
    path = [(x0, y0)]
    x, y = x0, y0
    s = np.zeros(2)
    for _ in range(n_steps):
        g = rosenbrock_grad(x, y)
        s = beta2 * s + (1 - beta2) * g ** 2
        x -= lr * g[0] / (np.sqrt(s[0]) + eps_opt)
        y -= lr * g[1] / (np.sqrt(s[1]) + eps_opt)
        path.append((x, y))
    return path

def run_adam(x0, y0, lr, beta1, beta2, n_steps, weight_decay=0.0):
    path = [(x0, y0)]
    x, y = x0, y0
    m = np.zeros(2)
    v = np.zeros(2)
    for t in range(1, n_steps + 1):
        g = rosenbrock_grad(x, y)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        update = lr * m_hat / (np.sqrt(v_hat) + eps_opt)
        x -= update[0] + lr * weight_decay * x
        y -= update[1] + lr * weight_decay * y
        path.append((x, y))
    return path

paths = {
    "SGD": run_sgd(x0, y0, lr_sgd, n_steps_opt),
    "Momentum": run_momentum(x0, y0, lr_momentum, beta_momentum, n_steps_opt),
    "RMSProp": run_rmsprop(x0, y0, lr_rmsprop, 0.99, n_steps_opt),
    "Adam": run_adam(x0, y0, lr_adam, 0.9, 0.999, n_steps_opt),
    "AdamW": run_adam(x0, y0, lr_adam, 0.9, 0.999, n_steps_opt, weight_decay=0.01),
}
colors_opt = {
    "SGD": "steelblue",
    "Momentum": "darkorange",
    "RMSProp": "green",
    "Adam": "red",
    "AdamW": "purple",
}

selected_opts = st.multiselect(
    "Оптимизаторы для отображения",
    list(paths.keys()),
    default=list(paths.keys()),
)

# Contour plot
x_grid = np.linspace(-2.0, 1.8, 300)
y_grid = np.linspace(-0.5, 3.5, 300)
X_g, Y_g = np.meshgrid(x_grid, y_grid)
Z_g = rosenbrock(X_g, Y_g)

fig, ax = plt.subplots(figsize=(10, 7))
levels = np.logspace(0, 3.5, 35)
cf = ax.contourf(X_g, Y_g, Z_g, levels=levels, cmap="viridis", alpha=0.6)
ax.contour(X_g, Y_g, Z_g, levels=levels, colors="white", linewidths=0.3, alpha=0.4)
plt.colorbar(cf, ax=ax, label="f(x,y) — log scale")

ax.scatter([1], [1], color="gold", s=200, zorder=5, marker="*", label="Минимум (1,1)")
ax.scatter([x0], [y0], color="white", s=100, zorder=5, marker="o", label="Старт")

for name in selected_opts:
    path = paths[name]
    xs, ys = zip(*path)
    ax.plot(xs, ys, color=colors_opt[name], lw=1.5, label=name, alpha=0.85)
    ax.scatter([xs[-1]], [ys[-1]], color=colors_opt[name], s=60, zorder=4, marker="D")

ax.set_xlim(-2.0, 1.8)
ax.set_ylim(-0.5, 3.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"Функция Розенброка — траектории оптимизаторов ({n_steps_opt} шагов)")
ax.legend(loc="upper left", fontsize=9)
st.pyplot(fig)

# Loss curves
fig2, ax2 = plt.subplots(figsize=(10, 4))
for name in selected_opts:
    path = paths[name]
    losses = [rosenbrock(p[0], p[1]) for p in path]
    ax2.semilogy(losses, color=colors_opt[name], lw=1.8, label=name)
ax2.set_xlabel("Шаг")
ax2.set_ylabel("f(x,y) — log scale")
ax2.set_title("Кривые сходимости (log scale)")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)

# Final values table
st.subheader("Финальное положение каждого оптимизатора")
rows = []
for name in selected_opts:
    xf, yf = paths[name][-1]
    loss_f = rosenbrock(xf, yf)
    dist = np.sqrt((xf - 1)**2 + (yf - 1)**2)
    rows.append({"Оптимизатор": name, "x": f"{xf:.4f}", "y": f"{yf:.4f}",
                 "f(x,y)": f"{loss_f:.4f}", "Расст. до (1,1)": f"{dist:.4f}"})

import pandas as pd
st.dataframe(pd.DataFrame(rows).set_index("Оптимизатор"), use_container_width=True)


st.header("Интерактивный шаг Adam — пошагово")

st.markdown(
    r"""
Проследим за обновлениями Adam на скалярной функции $f(\theta) = \theta^2$ (минимум в 0).

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t}+\varepsilon}\,\hat{m}_t$$
"""
)

col_a1, col_a2, col_a3 = st.columns(3)
with col_a1:
    theta_init = st.slider("Начальное θ₀", -5.0, 5.0, 3.0, 0.5)
    lr_adam_demo = st.slider("η (learning rate)", 0.01, 1.0, 0.1, 0.01)
with col_a2:
    b1_demo = st.slider("β₁", 0.0, 0.999, 0.9, 0.01)
    b2_demo = st.slider("β₂", 0.0, 0.9999, 0.999, 0.001)
with col_a3:
    n_steps_demo = st.slider("Число шагов", 1, 50, 20)

theta = theta_init
m_d, v_d = 0.0, 0.0
thetas = [theta]
for t in range(1, n_steps_demo + 1):
    g = 2 * theta  # grad of theta^2
    m_d = b1_demo * m_d + (1 - b1_demo) * g
    v_d = b2_demo * v_d + (1 - b2_demo) * g ** 2
    m_hat = m_d / (1 - b1_demo ** t)
    v_hat = v_d / (1 - b2_demo ** t)
    theta = theta - lr_adam_demo * m_hat / (np.sqrt(v_hat) + 1e-8)
    thetas.append(theta)

t_range = np.linspace(-5, 5, 200)
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))

axes3[0].plot(t_range, t_range ** 2, color="gray", lw=1.5, label="f(θ)=θ²")
axes3[0].scatter(thetas, [t ** 2 for t in thetas], c=range(len(thetas)),
                 cmap="plasma", zorder=5, s=40)
axes3[0].scatter([thetas[0]], [thetas[0]**2], color="green", s=100, zorder=6, marker="o", label="Старт")
axes3[0].scatter([thetas[-1]], [thetas[-1]**2], color="red", s=100, zorder=6, marker="D", label="Финиш")
axes3[0].set_title("Траектория θ по параболе f(θ)=θ²")
axes3[0].set_xlabel("θ")
axes3[0].set_ylabel("f(θ)")
axes3[0].legend()
axes3[0].grid(True, alpha=0.3)

axes3[1].plot(range(len(thetas)), [t ** 2 for t in thetas], color="steelblue", marker="o", ms=4)
axes3[1].set_yscale("log")
axes3[1].set_title("Сходимость Adam: f(θ) по шагам (log scale)")
axes3[1].set_xlabel("Шаг")
axes3[1].set_ylabel("f(θ) = θ²")
axes3[1].grid(True, alpha=0.3)

st.pyplot(fig3)
st.success(f"После {n_steps_demo} шагов: θ = {thetas[-1]:.6f}, f(θ) = {thetas[-1]**2:.8f}")
