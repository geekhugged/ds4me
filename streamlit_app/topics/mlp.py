import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("1. Перцептрон и многослойные нейронные сети (MLP)")
st.caption("Теория и формулы — pages/deep-learning.html, раздел 1.")

st.markdown(
    r"""
**MLP (Multilayer Perceptron)** — стек полносвязных слоёв с нелинейными активациями.

Прямой проход через слой $l$:

$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]}\,\mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}, \qquad \mathbf{a}^{[l]} = g\!\left(\mathbf{z}^{[l]}\right)$$

Ниже вы можете задать архитектуру сети и наблюдать, как входной вектор проходит сквозь слои.
"""
)

st.header("Интерактивный прямой проход")

col1, col2 = st.columns(2)
with col1:
    n_input = st.slider("Входных нейронов", 1, 8, 3)
    n_hidden1 = st.slider("Нейронов в скрытом слое 1", 1, 16, 4)
with col2:
    n_hidden2 = st.slider("Нейронов в скрытом слое 2", 0, 16, 3,
                           help="0 — использовать только один скрытый слой")
    n_output = st.slider("Выходных нейронов", 1, 8, 2)

activation_name = st.selectbox("Функция активации скрытых слоёв", ["ReLU", "Tanh", "Sigmoid"])

def activate(z, name):
    if name == "ReLU":
        return np.maximum(0, z)
    elif name == "Tanh":
        return np.tanh(z)
    else:
        return 1 / (1 + np.exp(-z))

rng = np.random.default_rng(42)

# Build layers
layers = [n_input, n_hidden1]
if n_hidden2 > 0:
    layers.append(n_hidden2)
layers.append(n_output)

# Initialize weights (Xavier)
weights = []
biases = []
for i in range(len(layers) - 1):
    scale = np.sqrt(2.0 / layers[i])
    W = rng.normal(0, scale, (layers[i + 1], layers[i]))
    b = np.zeros(layers[i + 1])
    weights.append(W)
    biases.append(b)

# Input vector
x = rng.uniform(-1, 1, n_input)

# Forward pass
activations = [x]
for idx, (W, b) in enumerate(zip(weights, biases)):
    z = W @ activations[-1] + b
    if idx < len(weights) - 1:
        a = activate(z, activation_name)
    else:
        a = z  # linear output
    activations.append(a)

# Visualize network
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis("off")

layer_names = (
    ["Вход"]
    + [f"Скрытый {i+1}" for i in range(len(layers) - 2)]
    + ["Выход"]
)

neuron_positions = []
for li, n in enumerate(layers):
    x_pos = li / (len(layers) - 1)
    y_positions = np.linspace(0.1, 0.9, n)
    neuron_positions.append([(x_pos, y) for y in y_positions])

# Draw connections
for li in range(len(layers) - 1):
    for src in neuron_positions[li]:
        for dst in neuron_positions[li + 1]:
            ax.plot([src[0], dst[0]], [src[1], dst[1]],
                    color="lightgray", linewidth=0.5, zorder=1)

# Color neurons by activation value
for li, (positions, act_vals) in enumerate(zip(neuron_positions, activations)):
    vals = act_vals if len(act_vals) == len(positions) else act_vals[:len(positions)]
    vmax = max(abs(vals).max(), 1e-6)
    for (xp, yp), v in zip(positions, vals):
        color_intensity = np.clip((v / vmax + 1) / 2, 0, 1)
        color = plt.cm.RdBu(color_intensity)
        circle = plt.Circle((xp, yp), 0.03, color=color, zorder=3, ec="black", lw=0.8)
        ax.add_patch(circle)
        ax.text(xp, yp, f"{v:.2f}", ha="center", va="center", fontsize=6, zorder=4)

# Layer labels
for li, (name, positions) in enumerate(zip(layer_names, neuron_positions)):
    ax.text(positions[0][0], 0.02, name, ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(0.0, 1.0)
ax.set_title(f"Прямой проход MLP: {' → '.join(str(n) for n in layers)}", fontsize=11)
st.pyplot(fig)

st.info(
    f"Цвет нейронов отражает значение активации (синий — отрицательное, красный — положительное). "
    f"Число параметров сети: **{sum(W.size + b.size for W, b in zip(weights, biases)):,}**."
)

st.header("Влияние ширины и глубины")

st.markdown(
    r"""
Следующий график показывает, как MLP аппроксимирует функцию $\sin(2\pi x)$
при разных архитектурах. Сеть обучается через псевдо-SGD (несколько итераций
numpy-градиентного спуска).
"""
)

col3, col4 = st.columns(2)
with col3:
    arch_width = st.slider("Ширина скрытого слоя", 2, 64, 16, 2)
with col4:
    arch_depth = st.slider("Глубина (кол-во скрытых слоёв)", 1, 4, 2)

n_steps = st.slider("Итераций обучения (SGD)", 100, 5000, 1000, 100)
lr_approx = st.slider("Learning rate (×0.001)", 1, 50, 10) * 0.001

if st.button("Запустить аппроксимацию"):
    x_train = np.linspace(0, 1, 100).reshape(-1, 1)
    y_train = np.sin(2 * np.pi * x_train)

    arch = [1] + [arch_width] * arch_depth + [1]
    rng2 = np.random.default_rng(0)

    Ws = [rng2.normal(0, np.sqrt(2 / arch[i]), (arch[i + 1], arch[i]))
          for i in range(len(arch) - 1)]
    Bs = [np.zeros((arch[i + 1], 1)) for i in range(len(arch) - 1)]

    def forward(x_in, Ws, Bs):
        a = x_in.T
        caches = [a]
        for idx, (W, b) in enumerate(zip(Ws, Bs)):
            z = W @ a + b
            a = np.maximum(0, z) if idx < len(Ws) - 1 else z
            caches.append(a)
        return caches

    def mse_loss(pred, target):
        return np.mean((pred - target.T) ** 2)

    losses = []
    for step in range(n_steps):
        caches = forward(x_train, Ws, Bs)
        pred = caches[-1]
        loss = mse_loss(pred, y_train)
        losses.append(loss)

        # Backprop
        dA = 2 * (pred - y_train.T) / x_train.shape[0]
        for idx in range(len(Ws) - 1, -1, -1):
            a_prev = caches[idx]
            if idx < len(Ws) - 1:
                z = Ws[idx] @ a_prev + Bs[idx]
                dZ = dA * (z > 0).astype(float)
            else:
                dZ = dA
            dW = dZ @ a_prev.T
            dB = dZ.sum(axis=1, keepdims=True)
            dA = Ws[idx].T @ dZ
            Ws[idx] -= lr_approx * dW
            Bs[idx] -= lr_approx * dB

    x_test = np.linspace(0, 1, 300).reshape(-1, 1)
    y_pred = forward(x_test, Ws, Bs)[-1].T

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(x_test, np.sin(2 * np.pi * x_test), label="sin(2πx)", color="gray", lw=2)
    ax1.plot(x_test, y_pred, label=f"MLP {arch}", color="steelblue", lw=2)
    ax1.scatter(x_train, y_train, color="gray", s=10, alpha=0.5)
    ax1.legend()
    ax1.set_title("Аппроксимация функции")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    ax2.semilogy(losses)
    ax2.set_title("Кривая обучения (MSE)")
    ax2.set_xlabel("Итерация")
    ax2.set_ylabel("Loss (log scale)")

    st.pyplot(fig2)
    st.success(f"Финальный MSE: **{losses[-1]:.6f}**")
else:
    st.info("Нажмите кнопку выше, чтобы запустить обучение MLP на numpy.")
