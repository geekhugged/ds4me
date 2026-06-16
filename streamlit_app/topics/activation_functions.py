import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.special import erf

st.title("2. Функции активации (Sigmoid, Tanh, ReLU, GELU, Softmax)")
st.caption("Теория и формулы — pages/deep-learning.html, раздел 2.")

st.markdown(
    r"""
Функция активации $g(z)$ вносит нелинейность в нейронную сеть. Без неё
$L$ линейных слоёв эквивалентны одному: $\mathbf{W}_L \cdots \mathbf{W}_1 \mathbf{x} = \mathbf{W}\mathbf{x}$.

| Функция | Формула | Диапазон | Проблема |
|---|---|---|---|
| Sigmoid | $\sigma(z)=\frac{1}{1+e^{-z}}$ | $(0,1)$ | Затухание градиента |
| Tanh | $\tanh(z)$ | $(-1,1)$ | Затухание градиента |
| ReLU | $\max(0,z)$ | $[0,+\infty)$ | Dying neurons |
| GELU | $z\cdot\Phi(z)$ | $\approx(-0.17,+\infty)$ | Дороже ReLU |
| Softmax | $\frac{e^{z_k}}{\sum e^{z_j}}$ | $(0,1)$, сумма=1 | Только для выхода |
"""
)

st.header("График функций и их производных")

z = np.linspace(-5, 5, 500)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_d(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh_d(z):
    return 1 - np.tanh(z) ** 2

def relu(z):
    return np.maximum(0, z)

def relu_d(z):
    return (z > 0).astype(float)

def gelu(z):
    return 0.5 * z * (1 + erf(z / np.sqrt(2)))

def gelu_d(z):
    phi = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    Phi = 0.5 * (1 + erf(z / np.sqrt(2)))
    return Phi + z * phi

def leaky_relu(z, alpha=0.1):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_d(z, alpha=0.1):
    return np.where(z > 0, 1.0, alpha)

functions = {
    "Sigmoid": (sigmoid, sigmoid_d, "steelblue"),
    "Tanh": (np.tanh, tanh_d, "darkorange"),
    "ReLU": (relu, relu_d, "green"),
    "GELU": (gelu, gelu_d, "purple"),
    "Leaky ReLU (α=0.1)": (leaky_relu, leaky_relu_d, "brown"),
}

selected = st.multiselect(
    "Выберите функции для отображения",
    list(functions.keys()),
    default=["Sigmoid", "Tanh", "ReLU", "GELU"],
)

show_derivatives = st.checkbox("Показать производные (пунктиром)", value=True)

fig, ax = plt.subplots(figsize=(10, 5))
ax.axhline(0, color="black", linewidth=0.7)
ax.axvline(0, color="black", linewidth=0.7)

for name in selected:
    fn, fn_d, color = functions[name]
    ax.plot(z, fn(z), color=color, lw=2, label=name)
    if show_derivatives:
        ax.plot(z, fn_d(z), color=color, lw=1.5, linestyle="--", alpha=0.6,
                label=f"{name}' (производная)")

ax.set_xlim(-5, 5)
ax.set_ylim(-1.5, 3.5)
ax.set_xlabel("z")
ax.set_ylabel("g(z)")
ax.set_title("Функции активации и их производные")
ax.legend(loc="upper left", fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
st.pyplot(fig)

st.header("Насыщение и затухание градиента")

st.markdown(
    r"""
При использовании Sigmoid или Tanh в глубоких сетях производная стремится к 0
при больших $|z|$. В цепном правиле обратного прохода эти малые числа
перемножаются: $\delta^{[1]} \propto \prod_{l} g'^{[l]}(z^{[l]}) \to 0$.
"""
)

depth = st.slider("Глубина сети (число слоёв)", 2, 20, 8)
act_for_vanish = st.radio(
    "Активация",
    ["Sigmoid", "Tanh", "ReLU"],
    horizontal=True,
)

z_val = st.slider("Входное значение z (одинаковое для всех слоёв)", -4.0, 4.0, 1.5, 0.1)

fn_map = {"Sigmoid": sigmoid_d, "Tanh": tanh_d, "ReLU": relu_d}
d_fn = fn_map[act_for_vanish]

grad_layers = [1.0]
for _ in range(depth):
    grad_layers.append(grad_layers[-1] * d_fn(z_val))

fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.semilogy(range(depth + 1), grad_layers, marker="o", color="steelblue")
ax2.set_xlabel("Слой (от выхода к входу)")
ax2.set_ylabel("Накопленный градиент (log scale)")
ax2.set_title(f"Затухание градиента: {act_for_vanish}, z={z_val:.1f}")
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)

last_grad = grad_layers[-1]
if last_grad < 1e-5:
    st.error(f"Градиент у входного слоя: {last_grad:.2e} — практически исчез (vanishing gradient)!")
elif last_grad > 1e3:
    st.error(f"Градиент у входного слоя: {last_grad:.2e} — взрыв градиента (exploding)!")
else:
    st.success(f"Градиент у входного слоя: {last_grad:.4f} — приемлемо.")

st.header("Softmax — нормировка вектора в вероятности")

st.markdown(
    r"""
$$\text{Softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$$

Измените логиты ниже и наблюдайте, как Softmax превращает их в вероятности.
"""
)

n_classes = st.slider("Число классов K", 2, 8, 4)
logits = []
cols = st.columns(n_classes)
for i, col in enumerate(cols):
    with col:
        logits.append(st.slider(f"z[{i}]", -5.0, 5.0, float(i - n_classes // 2), 0.5))

logits_arr = np.array(logits)
# Numerically stable softmax
logits_stable = logits_arr - logits_arr.max()
exp_z = np.exp(logits_stable)
probs = exp_z / exp_z.sum()

fig3, ax3 = plt.subplots(figsize=(7, 4))
bars = ax3.bar(range(n_classes), probs, color="steelblue", edgecolor="black")
ax3.set_xticks(range(n_classes))
ax3.set_xticklabels([f"Класс {i}" for i in range(n_classes)])
ax3.set_ylim(0, 1.05)
ax3.set_ylabel("Вероятность")
ax3.set_title("Softmax: логиты → вероятности")
for bar, p in zip(bars, probs):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{p:.3f}", ha="center", va="bottom", fontsize=9)
ax3.grid(True, axis="y", alpha=0.3)
st.pyplot(fig3)

st.write(f"Логиты: {[f'{v:.1f}' for v in logits_arr]} | "
         f"Вероятности: {[f'{p:.3f}' for p in probs]} | "
         f"Сумма: {probs.sum():.6f}")
