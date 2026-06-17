import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("6. Инициализация весов (Xavier/Glorot, He)")
st.caption("Теория и формулы — pages/deep-learning.html, раздел 6.")

st.markdown(
    r"""
От масштаба начальных весов зависит, сохраняется ли дисперсия сигнала при проходе через
много слоёв подряд. Если веса слишком велики — активации/градиенты взрываются; если слишком
малы — затухают. Правильная инициализация подбирает дисперсию весов так, чтобы это не
происходило.

$$\text{Var}(z) = n_{in}\,\sigma_w^2\,\sigma_x^2$$

- **Xavier/Glorot** (для tanh/sigmoid): $\sigma_w^2 = \dfrac{2}{n_{in}+n_{out}}$
- **He / Kaiming** (для ReLU и вариантов): $\sigma_w^2 = \dfrac{2}{n_{in}}$
"""
)

st.header("Распределение активаций по глубине сети")

st.markdown(
    """
Симулируем сеть из $L$ полносвязных слоёв без обучения (только forward pass) с заданной
схемой инициализации и активацией, и смотрим, как меняется распределение активаций
(и его стандартное отклонение) от слоя к слою.
"""
)

col1, col2, col3 = st.columns(3)
with col1:
    n_layers = st.slider("Число слоёв L", 2, 30, 15, 1)
    n_units = st.slider("Нейронов в слое", 16, 512, 128, 16)
with col2:
    activation_name = st.radio("Функция активации", ["tanh", "sigmoid", "ReLU"], horizontal=False)
    init_scheme = st.radio(
        "Схема инициализации",
        ["Xavier/Glorot", "He/Kaiming", "Слишком большая (×5)", "Слишком маленькая (÷5)", "Нули"],
    )
with col3:
    batch_size = st.slider("Размер батча (число примеров)", 64, 2000, 500, 64)
    seed_init = st.number_input("Random seed", 0, 9999, 42)

rng = np.random.default_rng(seed_init)

def activation_fn(name):
    if name == "tanh":
        return np.tanh, lambda a: 1 - a ** 2
    if name == "sigmoid":
        sig = lambda z: 1 / (1 + np.exp(-z))
        return sig, lambda a: a * (1 - a)
    # ReLU
    return lambda z: np.maximum(0, z), lambda a: (a > 0).astype(float)

act_fn, act_deriv = activation_fn(activation_name)

def weight_std(scheme, n_in, n_out):
    if scheme == "Xavier/Glorot":
        return np.sqrt(2.0 / (n_in + n_out))
    if scheme == "He/Kaiming":
        return np.sqrt(2.0 / n_in)
    if scheme == "Слишком большая (×5)":
        return 5 * np.sqrt(2.0 / (n_in + n_out))
    if scheme == "Слишком маленькая (÷5)":
        return np.sqrt(2.0 / (n_in + n_out)) / 5
    return 0.0  # Нули

X = rng.normal(size=(batch_size, n_units))  # imitate normalized input

activations = [X]
preactivations = []
weights_list = []
a = X
for l in range(n_layers):
    n_in = a.shape[1]
    n_out = n_units
    std = weight_std(init_scheme, n_in, n_out)
    W = rng.normal(0, std, size=(n_in, n_out)) if std > 0 else np.zeros((n_in, n_out))
    b = np.zeros(n_out)
    z = a @ W + b
    a = act_fn(z)
    preactivations.append(z)
    activations.append(a)
    weights_list.append(W)

means = [np.mean(a) for a in activations[1:]]
stds = [np.std(a) for a in activations[1:]]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# histograms for a few layers
layers_to_show = sorted(set([0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]))
for idx in layers_to_show:
    axes[0].hist(activations[idx + 1].flatten(), bins=60, alpha=0.5, density=True,
                 label=f"слой {idx+1}")
axes[0].set_title("Распределение активаций в выбранных слоях")
axes[0].set_xlabel("Значение активации")
axes[0].set_ylabel("Плотность")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, n_layers + 1), stds, marker="o", color="steelblue", label="std(активации)")
axes[1].set_yscale("log")
axes[1].set_xlabel("Номер слоя")
axes[1].set_ylabel("Стандартное отклонение (log scale)")
axes[1].set_title("Эволюция std активаций по глубине сети")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

st.pyplot(fig)

last_std = stds[-1]
if last_std < 1e-3:
    st.error(f"std активаций на последнем слое ≈ {last_std:.2e} — сигнал практически затух (vanishing).")
elif last_std > 10:
    st.error(f"std активаций на последнем слое ≈ {last_std:.2e} — сигнал взорвался (exploding).")
else:
    st.success(f"std активаций на последнем слое ≈ {last_std:.4f} — сигнал стабилен по глубине.")

st.header("Норма градиента по слоям при обратном проходе")

st.markdown(
    """
Смоделируем обратный проход: возьмём случайный градиент потери по выходу последнего слоя
и пропустим его назад через всю сеть (та же активация и веса), наблюдая норму градиента
на каждом слое.
"""
)

grad = rng.normal(size=activations[-1].shape)
grad_norms = [np.linalg.norm(grad)]
for l in range(n_layers - 1, -1, -1):
    da = act_deriv(activations[l + 1])
    delta = grad * da
    grad = delta @ weights_list[l].T
    grad_norms.append(np.linalg.norm(grad))

grad_norms = grad_norms[::-1]  # from input layer to output layer

fig2, ax2 = plt.subplots(figsize=(9, 4))
ax2.plot(range(len(grad_norms)), grad_norms, marker="o", color="darkorange")
ax2.set_yscale("log")
ax2.set_xlabel("Номер слоя (0 = вход)")
ax2.set_ylabel("‖градиент‖ (log scale)")
ax2.set_title(f"Норма градиента по слоям — {init_scheme}, активация {activation_name}")
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)

st.markdown(
    """
Попробуйте переключить схему инициализации на «Слишком большая» или «Слишком маленькая» —
норма градиента и std активаций начнут экспоненциально расти/затухать с глубиной. Xavier
для tanh/sigmoid и He для ReLU обычно дают наиболее стабильную (плоскую) кривую.
"""
)
