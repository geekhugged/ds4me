import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("8. Vanishing/exploding gradients и residual connections (ResNet)")
st.caption("Теория и формулы — pages/deep-learning.html, раздел 8.")

st.markdown(
    r"""
Градиент по активации раннего слоя — произведение якобианов всех промежуточных слоёв:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[1]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[L]}} \prod_{l=2}^{L} \mathbf{W}^{[l]\top}\,\text{diag}(g'(\mathbf{z}^{[l]}))$$

Если норма каждого множителя $\approx c$, итоговая норма градиента ведёт себя как $c^{L-1}$ —
экспоненциально затухает при $c<1$ и взрывается при $c>1$.

Residual block: $\mathbf{y} = \mathbf{x} + F(\mathbf{x})$. Градиент по входу блока:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}}\left(\mathbf{I} + \frac{\partial F}{\partial \mathbf{x}}\right)$$
Слагаемое $\mathbf{I}$ — «градиентное шоссе», которое не зависит от глубины.
"""
)

st.header("Норма градиента по слоям: plain network vs ResNet")

col1, col2, col3 = st.columns(3)
with col1:
    n_layers_vg = st.slider("Глубина сети L", 5, 100, 40, 5)
    width_vg = st.slider("Ширина слоя", 16, 128, 64, 16)
with col2:
    activation_vg = st.radio("Активация", ["tanh", "sigmoid", "ReLU"])
    weight_scale_vg = st.slider(
        "Масштаб инициализации весов (× от He/Xavier)", 0.2, 2.0, 1.0, 0.1
    )
with col3:
    use_residual = st.checkbox("Включить residual connections", value=True)
    seed_vg = st.number_input("Random seed", 0, 9999, 1)

rng = np.random.default_rng(seed_vg)

def act_pair(name):
    if name == "tanh":
        return np.tanh, lambda a: 1 - a ** 2
    if name == "sigmoid":
        s = lambda z: 1 / (1 + np.exp(-z))
        return s, lambda a: a * (1 - a)
    return lambda z: np.maximum(0, z), lambda a: (a > 0).astype(float)

act_fn, act_deriv = act_pair(activation_vg)
base_std = np.sqrt(2.0 / width_vg) if activation_vg == "ReLU" else np.sqrt(2.0 / (2 * width_vg))
std_vg = base_std * weight_scale_vg

def simulate_network(use_res):
    batch = 256
    X = rng.normal(size=(batch, width_vg))
    a = X
    weights, preacts, acts = [], [], [a]
    for _ in range(n_layers_vg):
        W = rng.normal(0, std_vg, size=(width_vg, width_vg))
        z = a @ W
        f = act_fn(z)
        a_next = a + f if use_res else f
        weights.append(W)
        preacts.append(z)
        acts.append(a_next)
        a = a_next
    # backward pass: random upstream grad
    grad = rng.normal(size=acts[-1].shape)
    grad_norms = [np.linalg.norm(grad)]
    for l in range(n_layers_vg - 1, -1, -1):
        da_f = act_deriv(act_fn(preacts[l]))
        dF_dz = da_f
        dz_da = weights[l]
        # d(F)/d(a_prev) = (dF/dz) @ W^T
        dF_da_prev = (grad * dF_dz) @ dz_da.T
        if use_res:
            grad = grad + dF_da_prev  # identity + branch
        else:
            grad = dF_da_prev
        grad_norms.append(np.linalg.norm(grad))
    grad_norms = grad_norms[::-1]
    act_stds = [np.std(a) for a in acts]
    return grad_norms, act_stds

grad_norms_res, stds_res = simulate_network(True)
grad_norms_plain, stds_plain = simulate_network(False)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(grad_norms_plain, marker="o", ms=3, color="firebrick", label="Plain network (без residual)")
axes[0].plot(grad_norms_res, marker="o", ms=3, color="steelblue", label="ResNet-style (с residual)")
axes[0].set_yscale("log")
axes[0].set_xlabel("Номер слоя (0 = вход)")
axes[0].set_ylabel("‖градиент‖ (log scale)")
axes[0].set_title(f"Норма градиента по слоям, активация={activation_vg}")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(stds_plain, marker="o", ms=3, color="firebrick", label="Plain network")
axes[1].plot(stds_res, marker="o", ms=3, color="steelblue", label="ResNet-style")
axes[1].set_yscale("log")
axes[1].set_xlabel("Номер слоя")
axes[1].set_ylabel("std(активации) (log scale)")
axes[1].set_title("Эволюция std активаций по глубине")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

st.pyplot(fig)

ratio = grad_norms_plain[0] / (grad_norms_plain[-1] + 1e-12)
ratio_res = grad_norms_res[0] / (grad_norms_res[-1] + 1e-12)
st.markdown(
    f"""
**Отношение нормы градиента (выход / вход) для plain-сети:** {1/ratio:.3e}
(если ≪1 — vanishing, если ≫1 — exploding)

**То же для ResNet-style сети:** {1/ratio_res:.3e}
"""
)
st.info(
    "Попробуйте увеличить глубину L до 60–100 с sigmoid/tanh без residual — норма "
    "градиента у входного слоя устремится к нулю (vanishing gradients). Включение "
    "residual connections почти всегда стабилизирует норму градиента независимо от глубины."
)

st.header("Gradient clipping: визуализация")

st.markdown(r"$$\mathbf{g} \leftarrow \mathbf{g}\cdot\min\!\left(1,\frac{\theta_{\max}}{\|\mathbf{g}\|_2}\right)$$")

theta_max = st.slider("Порог clipping θ_max", 0.5, 10.0, 3.0, 0.5)
raw_grad_norms = np.array(grad_norms_plain)
clipped = np.minimum(raw_grad_norms, theta_max)

fig2, ax2 = plt.subplots(figsize=(9, 4))
ax2.plot(raw_grad_norms, color="firebrick", marker="o", ms=3, label="Без clipping")
ax2.plot(clipped, color="green", marker="o", ms=3, label=f"С clipping (θ_max={theta_max})")
ax2.axhline(theta_max, color="gray", ls="--", lw=1)
ax2.set_xlabel("Номер слоя")
ax2.set_ylabel("‖градиент‖")
ax2.set_title("Эффект gradient clipping по норме")
ax2.legend()
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)
