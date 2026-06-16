import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

st.title("3. Backpropagation — прямое и обратное распространение ошибки")
st.caption("Теория и формулы — pages/deep-learning.html, раздел 3.")

st.markdown(
    r"""
Backpropagation — эффективный алгоритм вычисления градиентов $\partial\mathcal{L}/\partial\mathbf{W}^{[l]}$
через цепное правило. Алгоритм работает в два прохода:

1. **Forward pass**: вычисляем $\mathbf{z}^{[l]} = \mathbf{W}^{[l]}\mathbf{a}^{[l-1]}+\mathbf{b}^{[l]}$
   и $\mathbf{a}^{[l]} = g(\mathbf{z}^{[l]})$ — сохраняем все промежуточные значения.
2. **Backward pass**: вычисляем ошибки слоёв $\boldsymbol{\delta}^{[l]}$ от выхода к входу:

$$\boldsymbol{\delta}^{[l]} = \left(\mathbf{W}^{[l+1]\top}\boldsymbol{\delta}^{[l+1]}\right) \odot g'\!\left(\mathbf{z}^{[l]}\right)$$

Градиенты параметров:
$$\frac{\partial\mathcal{L}}{\partial\mathbf{W}^{[l]}} = \boldsymbol{\delta}^{[l]}\,\mathbf{a}^{[l-1]\top}, \qquad \frac{\partial\mathcal{L}}{\partial\mathbf{b}^{[l]}} = \boldsymbol{\delta}^{[l]}$$
"""
)

st.header("Пошаговая визуализация backprop на малой сети")

st.markdown(
    "Сеть: **1 → 2 → 1** (один вход, два нейрона в скрытом слое, один выход). "
    "Активация скрытого слоя — Sigmoid, выходной слой — линейный. Потеря — MSE."
)

col1, col2 = st.columns(2)
with col1:
    x_val = st.slider("Вход x", -3.0, 3.0, 1.0, 0.1)
    y_true_val = st.slider("Истинный ответ y", -3.0, 3.0, 0.5, 0.1)
with col2:
    w1_val = st.slider("W[1][0,0] (нейрон 0 ← вход)", -3.0, 3.0, 0.8, 0.1)
    w2_val = st.slider("W[1][1,0] (нейрон 1 ← вход)", -3.0, 3.0, -0.5, 0.1)
    w3_val = st.slider("W[2][0,0] (выход ← нейрон 0)", -3.0, 3.0, 1.2, 0.1)
    w4_val = st.slider("W[2][0,1] (выход ← нейрон 1)", -3.0, 3.0, -0.9, 0.1)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))

def sigmoid_d(z):
    s = sigmoid(z)
    return s * (1 - s)

# Network parameters
W1 = np.array([[w1_val], [w2_val]])   # shape (2,1)
b1 = np.array([0.0, 0.0])
W2 = np.array([[w3_val, w4_val]])      # shape (1,2)
b2 = np.array([0.0])

x = np.array([x_val])
y = np.array([y_true_val])

# Forward pass
z1 = W1 @ x + b1          # (2,)
a1 = sigmoid(z1)           # (2,)
z2 = W2 @ a1 + b2         # (1,)
a2 = z2                    # linear output
loss = 0.5 * (a2[0] - y[0]) ** 2

# Backward pass
dL_da2 = a2 - y           # (1,)
dL_dz2 = dL_da2            # linear activation => g'=1
dL_dW2 = np.outer(dL_dz2, a1)  # (1,2)
dL_db2 = dL_dz2

dL_da1 = W2.T @ dL_dz2    # (2,)
dL_dz1 = dL_da1 * sigmoid_d(z1)   # (2,)
dL_dW1 = np.outer(dL_dz1, x)      # (2,1)
dL_db1 = dL_dz1

# Display computation trace
steps = [
    ("FORWARD PASS", None),
    ("z₁ = W₁·x + b₁", f"[{w1_val:.2f}·{x_val:.2f}, {w2_val:.2f}·{x_val:.2f}] = [{z1[0]:.4f}, {z1[1]:.4f}]"),
    ("a₁ = σ(z₁)", f"[σ({z1[0]:.4f}), σ({z1[1]:.4f})] = [{a1[0]:.4f}, {a1[1]:.4f}]"),
    ("z₂ = W₂·a₁ + b₂", f"{w3_val:.2f}·{a1[0]:.4f} + {w4_val:.2f}·{a1[1]:.4f} = {z2[0]:.4f}"),
    ("ŷ = a₂ = z₂ (линейный)", f"{a2[0]:.4f}"),
    ("L = ½(ŷ - y)² ", f"½({a2[0]:.4f} - {y[0]:.2f})² = {loss:.6f}"),
    ("BACKWARD PASS", None),
    ("∂L/∂a₂ = ŷ - y", f"{dL_da2[0]:.6f}"),
    ("∂L/∂z₂ = ∂L/∂a₂ (g'=1)", f"{dL_dz2[0]:.6f}"),
    ("∂L/∂W₂ = δ₂ · a₁ᵀ", f"[{dL_dW2[0,0]:.6f}, {dL_dW2[0,1]:.6f}]"),
    ("∂L/∂a₁ = W₂ᵀ · δ₂", f"[{dL_da1[0]:.6f}, {dL_da1[1]:.6f}]"),
    ("σ'(z₁) = a₁(1-a₁)", f"[{sigmoid_d(z1)[0]:.6f}, {sigmoid_d(z1)[1]:.6f}]"),
    ("∂L/∂z₁ = ∂L/∂a₁ ⊙ σ'(z₁)", f"[{dL_dz1[0]:.6f}, {dL_dz1[1]:.6f}]"),
    ("∂L/∂W₁ = δ₁ · xᵀ", f"[[{dL_dW1[0,0]:.6f}], [{dL_dW1[1,0]:.6f}]]"),
]

st.subheader("Трассировка вычислений")
for label, value in steps:
    if value is None:
        st.markdown(f"**--- {label} ---**")
    else:
        col_l, col_r = st.columns([2, 3])
        col_l.code(label)
        col_r.write(value)

st.subheader("Визуализация сети с градиентами")

fig, ax = plt.subplots(figsize=(10, 5))
ax.axis("off")

neuron_coords = {
    "x": (0.1, 0.5),
    "h0": (0.45, 0.72),
    "h1": (0.45, 0.28),
    "out": (0.8, 0.5),
}

grad_magnitudes = {
    "W1_0": abs(dL_dW1[0, 0]),
    "W1_1": abs(dL_dW1[1, 0]),
    "W2_0": abs(dL_dW2[0, 0]),
    "W2_1": abs(dL_dW2[0, 1]),
}
gmax = max(grad_magnitudes.values()) + 1e-9

edges = [
    ("x", "h0", w1_val, dL_dW1[0, 0], "W1[0,0]"),
    ("x", "h1", w2_val, dL_dW1[1, 0], "W1[1,0]"),
    ("h0", "out", w3_val, dL_dW2[0, 0], "W2[0,0]"),
    ("h1", "out", w4_val, dL_dW2[0, 1], "W2[0,1]"),
]

for src, dst, w, g, label in edges:
    xs, ys = neuron_coords[src]
    xd, yd = neuron_coords[dst]
    lw = 1 + 4 * abs(g) / gmax
    color = "red" if g > 0 else "blue"
    ax.annotate("", xy=(xd, yd), xytext=(xs, ys),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw))
    mx, my = (xs + xd) / 2, (ys + yd) / 2
    ax.text(mx, my + 0.04, f"w={w:.2f}\n∂={g:.3f}", ha="center", va="bottom",
            fontsize=7, color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

node_labels = {
    "x": f"x\n{x_val:.2f}",
    "h0": f"h₀\na={a1[0]:.3f}",
    "h1": f"h₁\na={a1[1]:.3f}",
    "out": f"ŷ\n{a2[0]:.3f}",
}
node_colors = {"x": "lightyellow", "h0": "lightblue", "h1": "lightblue", "out": "lightgreen"}

for name, (xp, yp) in neuron_coords.items():
    circle = plt.Circle((xp, yp), 0.07, color=node_colors[name], ec="black", lw=1.5, zorder=3)
    ax.add_patch(circle)
    ax.text(xp, yp, node_labels[name], ha="center", va="center", fontsize=8, zorder=4)

ax.text(0.5, 0.95, f"Loss = {loss:.6f}  |  y_true = {y_true_val:.2f}",
        ha="center", va="top", transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round", fc="lightyellow", ec="orange"))

red_patch = mpatches.Patch(color="red", label="∂L/∂w > 0 (уменьшить w)")
blue_patch = mpatches.Patch(color="blue", label="∂L/∂w < 0 (увеличить w)")
ax.legend(handles=[red_patch, blue_patch], loc="lower center",
          bbox_to_anchor=(0.5, 0.0), fontsize=8)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("Градиенты в сети 1→2→1 (толщина стрелки = |∂L/∂w|)", fontsize=11)
st.pyplot(fig)

st.header("Проверка градиентов (Gradient Checking)")

st.markdown(
    r"""
Численная проверка градиента методом конечных разностей:
$$\frac{\partial \mathcal{L}}{\partial \theta} \approx \frac{\mathcal{L}(\theta+\varepsilon) - \mathcal{L}(\theta-\varepsilon)}{2\varepsilon}$$
"""
)

eps = st.select_slider("Эпсилон ε для проверки", options=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5], value=1e-4)

def compute_loss(W1, b1, W2, b2, x, y):
    z1 = W1 @ x + b1
    a1 = sigmoid(z1)
    z2 = W2 @ a1 + b2
    return 0.5 * (z2[0] - y[0]) ** 2

analytic_grad = dL_dW2[0, 0]

W2_plus = W2.copy(); W2_plus[0, 0] += eps
W2_minus = W2.copy(); W2_minus[0, 0] -= eps
numeric_grad = (compute_loss(W1, b1, W2_plus, b2, x, y) -
                compute_loss(W1, b1, W2_minus, b2, x, y)) / (2 * eps)

rel_error = abs(analytic_grad - numeric_grad) / (abs(analytic_grad) + abs(numeric_grad) + 1e-10)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Аналитический градиент ∂L/∂W₂[0,0]", f"{analytic_grad:.8f}")
col_b.metric("Численный градиент (конечные разности)", f"{numeric_grad:.8f}")
col_c.metric("Относительная ошибка", f"{rel_error:.2e}")

if rel_error < 1e-5:
    st.success("Градиент верен! Относительная ошибка < 1e-5.")
elif rel_error < 1e-3:
    st.warning(f"Небольшое расхождение (ε слишком велик или слишком мал?).")
else:
    st.error("Большое расхождение — возможна ошибка в реализации backprop.")
