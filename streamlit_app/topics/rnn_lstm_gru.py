import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("10. Рекуррентные сети: RNN, LSTM, GRU и проблема затухающих/взрывающихся градиентов")
st.caption("Теория и формулы — pages/deep-learning.html, раздел 10.")

st.markdown(
    r"""
Vanilla RNN: $h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$ — одни и те же веса разделяются
на каждом временном шаге.

Градиент по раннему скрытому состоянию — произведение якобианов по всем промежуточным шагам:
$$\frac{\partial \mathcal{L}_T}{\partial h_k} = \frac{\partial \mathcal{L}_T}{\partial h_T}\prod_{t=k+1}^{T} W_{hh}^\top\,\text{diag}(1-\tanh^2(z_t))$$

LSTM вводит память $c_t$ с **аддитивным** обновлением через гейты:
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
При $f_t \approx 1$ градиент по $c_{t-1}$ проходит почти без затухания — «градиентное шоссе».
"""
)

st.header("Развёрнутая во времени сеть (unrolling)")

T_unroll = st.slider("Длина последовательности для визуализации", 3, 12, 6)

fig0, ax0 = plt.subplots(figsize=(12, 2.5))
xs = np.arange(T_unroll)
for t in xs:
    ax0.scatter(t, 1, s=900, color="steelblue", zorder=3)
    ax0.text(t, 1, f"$h_{{{t}}}$", ha="center", va="center", color="white", fontsize=11, zorder=4)
    ax0.scatter(t, 0, s=600, color="lightgray", zorder=3)
    ax0.text(t, 0, f"$x_{{{t}}}$", ha="center", va="center", fontsize=10, zorder=4)
    ax0.annotate("", xy=(t, 0.85), xytext=(t, 0.15),
                 arrowprops=dict(arrowstyle="->", color="gray"))
    if t > 0:
        ax0.annotate("", xy=(t - 0.15, 1), xytext=(t - 0.85, 1),
                     arrowprops=dict(arrowstyle="->", color="darkorange", lw=2))
ax0.set_xlim(-0.7, T_unroll - 0.3)
ax0.set_ylim(-0.5, 1.5)
ax0.axis("off")
ax0.set_title("RNN, развёрнутая по времени: одни и те же веса W_hh, W_xh применяются на каждом шаге")
st.pyplot(fig0)

st.header("Норма градиента по времени: vanilla RNN vs LSTM")

st.markdown(
    "Реализуем forward/backward (BPTT) для vanilla RNN и для LSTM на длинной "
    "последовательности со случайными входами, и сравним норму градиента $\\partial \\mathcal{L}/\\partial h_k$ "
    "по шагам $k$."
)

col1, col2, col3 = st.columns(3)
with col1:
    seq_len = st.slider("Длина последовательности T", 10, 100, 50, 5)
    hidden_size = st.slider("Размер скрытого состояния", 4, 32, 8, 4)
with col2:
    w_scale = st.slider("Масштаб весов W_hh (vanilla RNN)", 0.1, 2.0, 1.0, 0.05)
    forget_bias = st.slider("Forget gate bias (LSTM)", -2.0, 3.0, 1.0, 0.5)
with col3:
    seed_rnn = st.number_input("Random seed", 0, 9999, 3)

rng = np.random.default_rng(seed_rnn)
input_size = hidden_size

# ---------- Vanilla RNN ----------
Whh = rng.normal(0, w_scale / np.sqrt(hidden_size), size=(hidden_size, hidden_size))
Wxh = rng.normal(0, 1 / np.sqrt(input_size), size=(input_size, hidden_size))
xs_seq = rng.normal(size=(seq_len, input_size))

h = np.zeros(hidden_size)
hs, zs = [h], []
for t in range(seq_len):
    z = h @ Whh + xs_seq[t] @ Wxh
    h = np.tanh(z)
    zs.append(z)
    hs.append(h)

# backward: random grad at last step, propagate back
grad_h = rng.normal(size=hidden_size)
rnn_grad_norms = [np.linalg.norm(grad_h)]
for t in range(seq_len - 1, -1, -1):
    dtanh = 1 - hs[t + 1] ** 2
    grad_z = grad_h * dtanh
    grad_h = grad_z @ Whh.T
    rnn_grad_norms.append(np.linalg.norm(grad_h))
rnn_grad_norms = rnn_grad_norms[::-1]

# ---------- LSTM ----------
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

scale_lstm = 1.0 / np.sqrt(hidden_size + input_size)
Wf = rng.normal(0, scale_lstm, size=(hidden_size + input_size, hidden_size))
Wi = rng.normal(0, scale_lstm, size=(hidden_size + input_size, hidden_size))
Wo = rng.normal(0, scale_lstm, size=(hidden_size + input_size, hidden_size))
Wc = rng.normal(0, scale_lstm, size=(hidden_size + input_size, hidden_size))
bf = np.full(hidden_size, forget_bias)

h_l = np.zeros(hidden_size)
c_l = np.zeros(hidden_size)
cache = []
cs, hs_l = [c_l], [h_l]
for t in range(seq_len):
    concat = np.concatenate([h_l, xs_seq[t]])
    f_t = sigmoid(concat @ Wf + bf)
    i_t = sigmoid(concat @ Wi)
    o_t = sigmoid(concat @ Wo)
    c_tilde = np.tanh(concat @ Wc)
    c_l = f_t * c_l + i_t * c_tilde
    h_l = o_t * np.tanh(c_l)
    cache.append((f_t, i_t, o_t, c_tilde, c_l.copy(), h_l.copy()))
    cs.append(c_l.copy())
    hs_l.append(h_l.copy())

grad_c = rng.normal(size=hidden_size)
lstm_grad_norms = [np.linalg.norm(grad_c)]
for t in range(seq_len - 1, -1, -1):
    f_t = cache[t][0]
    # simplified BPTT: gradient through cell state path only (dominant "highway" term)
    grad_c = grad_c * f_t
    lstm_grad_norms.append(np.linalg.norm(grad_c))
lstm_grad_norms = lstm_grad_norms[::-1]

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(rnn_grad_norms, color="firebrick", marker="o", ms=3, label="Vanilla RNN: ‖∂h_T/∂h_k‖")
ax.plot(lstm_grad_norms, color="steelblue", marker="o", ms=3, label="LSTM: ‖∂c_T/∂c_k‖ (cell-путь)")
ax.set_yscale("log")
ax.set_xlabel("Временной шаг k")
ax.set_ylabel("Норма градиента (log scale)")
ax.set_title(f"Затухание градиента по времени, T={seq_len}")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

ratio_rnn = rnn_grad_norms[0] / (rnn_grad_norms[-1] + 1e-12)
ratio_lstm = lstm_grad_norms[0] / (lstm_grad_norms[-1] + 1e-12)

c1, c2 = st.columns(2)
with c1:
    st.metric("Vanilla RNN: град. на шаге 0 / град. на шаге T", f"{ratio_rnn:.2e}")
with c2:
    st.metric("LSTM (cell-путь): град. на шаге 0 / град. на шаге T", f"{ratio_lstm:.2e}")

st.info(
    "При w_scale > 1 (норма W_hh выше 1) и достаточно длинной последовательности vanilla "
    "RNN-градиент часто взрывается; при w_scale < 1 — затухает почти до нуля уже за "
    "10–20 шагов. LSTM с положительным forget gate bias (f_t близко к 1) сохраняет "
    "градиент по cell-state почти на одном уровне на десятки и сотни шагов — это и есть "
    "основная причина, почему LSTM учит длинные зависимости лучше vanilla RNN."
)

st.header("Гейты LSTM на одном шаге — интуиция")

st.markdown(
    r"""
$$f_t=\sigma(W_f[h_{t-1},x_t]+b_f),\quad i_t=\sigma(W_i[h_{t-1},x_t]+b_i),\quad o_t=\sigma(W_o[h_{t-1},x_t]+b_o)$$
$$\tilde c_t=\tanh(W_c[h_{t-1},x_t]+b_c),\quad c_t=f_t\odot c_{t-1}+i_t\odot \tilde c_t,\quad h_t=o_t\odot\tanh(c_t)$$
"""
)

step_idx = st.slider("Временной шаг для просмотра значений гейтов", 0, seq_len - 1, 0)
f_v, i_v, o_v, c_tilde_v, c_v, h_v = cache[step_idx]

fig2, ax2 = plt.subplots(figsize=(9, 3.5))
labels = ["forget gate f_t", "input gate i_t", "output gate o_t"]
values = [f_v.mean(), i_v.mean(), o_v.mean()]
ax2.bar(labels, values, color=["firebrick", "steelblue", "green"])
ax2.set_ylim(0, 1)
ax2.set_ylabel("Среднее значение гейта по компонентам скрытого состояния")
ax2.set_title(f"Средние значения гейтов на шаге t={step_idx}")
ax2.grid(True, axis="y", alpha=0.3)
st.pyplot(fig2)
