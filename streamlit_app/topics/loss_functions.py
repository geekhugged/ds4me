import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("4. Функции потерь (MSE, Cross-Entropy, KL-Divergence)")
st.caption("Теория и формулы — pages/deep-learning.html, раздел 4.")

st.markdown(
    r"""
Функция потерь $\mathcal{L}(\hat{y}, y)$ задаёт, **что именно** оптимизирует модель.
Выбор потери вытекает из статистических предположений о задаче:

- **MSE** — предположение о гауссовом шуме в регрессии
- **Cross-Entropy** — предположение о распределении Бернулли / Categorical
- **KL-Divergence** — мера «расстояния» между распределениями, используется в VAE,
  дистилляции знаний, RLHF
"""
)

tab1, tab2, tab3 = st.tabs(["MSE", "Cross-Entropy", "KL-Divergence"])

# ========== TAB 1: MSE ==========
with tab1:
    st.subheader("Mean Squared Error (MSE)")
    st.markdown(
        r"""
$$\mathcal{L}_\text{MSE} = \frac{1}{m}\sum_{i=1}^{m}\left(\hat{y}^{(i)} - y^{(i)}\right)^2$$

MSE сильно штрафует выбросы (штраф растёт квадратично). Альтернативы:
- **MAE** $= \frac{1}{m}\sum|y - \hat{y}|$ — устойчива к выбросам, но не дифференцируема в 0.
- **Huber Loss** — MSE при малых ошибках, MAE при больших.
"""
    )

    y_range = st.slider("Диапазон истинного значения y", -5.0, 5.0, 0.0, 0.5)
    delta_huber = st.slider("Параметр δ для Huber Loss", 0.1, 5.0, 1.0, 0.1)

    errors = np.linspace(-6, 6, 400)
    mse_vals = errors ** 2
    mae_vals = np.abs(errors)
    huber_vals = np.where(
        np.abs(errors) <= delta_huber,
        0.5 * errors ** 2,
        delta_huber * (np.abs(errors) - 0.5 * delta_huber),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(errors, mse_vals, label="MSE", lw=2, color="steelblue")
    axes[0].plot(errors, mae_vals, label="MAE", lw=2, color="darkorange")
    axes[0].plot(errors, huber_vals, label=f"Huber (δ={delta_huber:.1f})", lw=2, color="green")
    axes[0].set_xlabel("Ошибка предсказания (ŷ - y)")
    axes[0].set_ylabel("Потеря")
    axes[0].set_title("Сравнение потерь для регрессии")
    axes[0].legend()
    axes[0].set_ylim(0, 20)
    axes[0].grid(True, alpha=0.3)

    # MSE vs outliers demo
    rng = np.random.default_rng(7)
    y_clean = rng.normal(0, 1, 50)
    outlier_strength = st.slider("Сила выброса (outlier)", 0.0, 20.0, 5.0, 0.5)
    y_with_outlier = np.append(y_clean, [outlier_strength])
    pred = np.zeros(len(y_with_outlier))

    mse_c = np.mean(y_clean ** 2)
    mse_o = np.mean(y_with_outlier ** 2)
    mae_c = np.mean(np.abs(y_clean))
    mae_o = np.mean(np.abs(y_with_outlier))

    categories = ["MSE без выброса", "MSE с выбросом", "MAE без выброса", "MAE с выбросом"]
    values = [mse_c, mse_o, mae_c, mae_o]
    colors_bar = ["steelblue", "red", "darkorange", "salmon"]

    axes[1].bar(categories, values, color=colors_bar)
    axes[1].set_title("Влияние выброса на MSE vs MAE")
    axes[1].set_ylabel("Значение потери")
    axes[1].tick_params(axis="x", rotation=15)
    for i, v in enumerate(values):
        axes[1].text(i, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    axes[1].grid(True, axis="y", alpha=0.3)

    st.pyplot(fig)
    st.info(f"Выброс силой {outlier_strength:.1f} увеличивает MSE в "
            f"**{mse_o/mse_c:.1f}x** раз, а MAE только в **{mae_o/mae_c:.1f}x** раз.")


# ========== TAB 2: Cross-Entropy ==========
with tab2:
    st.subheader("Cross-Entropy (Бинарная и Многоклассовая)")
    st.markdown(
        r"""
**Binary Cross-Entropy** ($y \in \{0,1\}$, $\hat{y} \in (0,1)$):
$$\mathcal{L}_\text{BCE} = -\left[y\log\hat{y} + (1-y)\log(1-\hat{y})\right]$$

**Categorical Cross-Entropy** ($K$ классов, $y_k \in \{0,1\}$ — one-hot):
$$\mathcal{L}_\text{CE} = -\sum_{k=1}^{K} y_k \log \hat{y}_k$$
"""
    )

    col1, col2 = st.columns(2)
    with col1:
        y_true_ce = st.radio("Истинный класс y", [0, 1], horizontal=True)
    with col2:
        p_hat = st.slider("Предсказанная вероятность ŷ класса 1", 0.001, 0.999, 0.7, 0.001)

    bce = -(y_true_ce * np.log(p_hat) + (1 - y_true_ce) * np.log(1 - p_hat))

    p_range = np.linspace(0.001, 0.999, 300)
    loss_y1 = -np.log(p_range)
    loss_y0 = -np.log(1 - p_range)

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

    axes2[0].plot(p_range, loss_y1, label="BCE при y=1: −log(ŷ)", color="steelblue", lw=2)
    axes2[0].plot(p_range, loss_y0, label="BCE при y=0: −log(1−ŷ)", color="darkorange", lw=2)
    current_curve = loss_y1 if y_true_ce == 1 else loss_y0
    axes2[0].scatter([p_hat], [bce], color="red", zorder=5, s=80,
                     label=f"Текущая точка: BCE={bce:.3f}")
    axes2[0].set_ylim(0, 8)
    axes2[0].set_xlabel("ŷ (предсказанная вероятность класса 1)")
    axes2[0].set_ylabel("Binary Cross-Entropy")
    axes2[0].set_title("Binary Cross-Entropy")
    axes2[0].legend(fontsize=8)
    axes2[0].grid(True, alpha=0.3)

    # Calibration
    p_grid = np.linspace(0.001, 0.999, 200)
    bce_grid = -(y_true_ce * np.log(p_grid) + (1 - y_true_ce) * np.log(1 - p_grid))

    axes2[1].plot(p_grid, bce_grid, lw=2, color="purple")
    axes2[1].axvline(p_hat, color="red", linestyle="--", label=f"ŷ={p_hat:.3f}")
    axes2[1].axhline(bce, color="gray", linestyle="--", label=f"BCE={bce:.3f}")
    axes2[1].scatter([p_hat], [bce], color="red", zorder=5, s=80)
    axes2[1].set_xlabel("ŷ")
    axes2[1].set_ylabel("BCE")
    axes2[1].set_title(f"BCE при y={y_true_ce} — штраф за уверенность в ошибке")
    axes2[1].legend(fontsize=8)
    axes2[1].grid(True, alpha=0.3)

    st.pyplot(fig2)
    st.metric("Binary Cross-Entropy", f"{bce:.4f}")

    st.markdown("---")
    st.markdown("**Многоклассовая CE: пример с K=4 классами**")

    K = 4
    true_class = st.radio("Истинный класс (0-индексация)", list(range(K)), horizontal=True)
    logits_ce = []
    cols_k = st.columns(K)
    for i, c in enumerate(cols_k):
        with c:
            logits_ce.append(c.slider(f"Логит z[{i}]", -5.0, 5.0, float(i == true_class) * 3 - 1.0, 0.5))
    logits_arr = np.array(logits_ce)
    exp_z = np.exp(logits_arr - logits_arr.max())
    probs_k = exp_z / exp_z.sum()
    ce_loss = -np.log(probs_k[true_class] + 1e-12)

    fig3, ax3 = plt.subplots(figsize=(6, 3))
    colors_k = ["lightblue"] * K
    colors_k[true_class] = "steelblue"
    bars = ax3.bar(range(K), probs_k, color=colors_k, edgecolor="black")
    for i, (bar, p) in enumerate(zip(bars, probs_k)):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{p:.3f}", ha="center", va="bottom", fontsize=9)
    ax3.set_xticks(range(K))
    ax3.set_xticklabels([f"Класс {i}" + (" ✓" if i == true_class else "") for i in range(K)])
    ax3.set_ylim(0, 1.15)
    ax3.set_ylabel("Вероятность (Softmax)")
    ax3.set_title(f"CE Loss = {ce_loss:.4f}  |  P(истинный класс {true_class}) = {probs_k[true_class]:.4f}")
    ax3.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig3)


# ========== TAB 3: KL-Divergence ==========
with tab3:
    st.subheader("KL-Divergence")
    st.markdown(
        r"""
KL-дивергенция измеряет «расстояние» от $Q$ до $P$:

$$D_\text{KL}(P \| Q) = \sum_x P(x)\,\log\frac{P(x)}{Q(x)}$$

**Свойства:** всегда $\geq 0$; $D_\text{KL}(P\|Q) = 0 \Leftrightarrow P = Q$; асимметрична:
$D_\text{KL}(P\|Q) \neq D_\text{KL}(Q\|P)$.

**Связь с CE:**
$$H(P, Q) = H(P) + D_\text{KL}(P \| Q)$$
Поэтому минимизировать CE по $Q$ эквивалентно минимизировать $D_\text{KL}(P\|Q)$.
"""
    )

    st.markdown("**Пример: KL между двумя гауссианами**")

    col_p, col_q = st.columns(2)
    with col_p:
        mu_p = st.slider("μ_P (среднее истинного распред.)", -3.0, 3.0, 0.0, 0.1)
        sigma_p = st.slider("σ_P (ст. откл. истинного)", 0.1, 3.0, 1.0, 0.1)
    with col_q:
        mu_q = st.slider("μ_Q (среднее модельного распред.)", -3.0, 3.0, 1.0, 0.1)
        sigma_q = st.slider("σ_Q (ст. откл. модельного)", 0.1, 3.0, 1.5, 0.1)

    # KL(P||Q) for Gaussians: closed form
    kl_pq = np.log(sigma_q / sigma_p) + (sigma_p**2 + (mu_p - mu_q)**2) / (2 * sigma_q**2) - 0.5
    kl_qp = np.log(sigma_p / sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 * sigma_p**2) - 0.5

    x_range = np.linspace(-8, 8, 500)
    p_vals = np.exp(-0.5 * ((x_range - mu_p) / sigma_p) ** 2) / (sigma_p * np.sqrt(2 * np.pi))
    q_vals = np.exp(-0.5 * ((x_range - mu_q) / sigma_q) ** 2) / (sigma_q * np.sqrt(2 * np.pi))

    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 4))

    axes4[0].fill_between(x_range, p_vals, alpha=0.4, color="steelblue", label=f"P = N({mu_p:.1f}, {sigma_p:.1f}²)")
    axes4[0].fill_between(x_range, q_vals, alpha=0.4, color="darkorange", label=f"Q = N({mu_q:.1f}, {sigma_q:.1f}²)")
    axes4[0].plot(x_range, p_vals, color="steelblue", lw=2)
    axes4[0].plot(x_range, q_vals, color="darkorange", lw=2)
    axes4[0].set_xlabel("x")
    axes4[0].set_ylabel("Плотность")
    axes4[0].set_title("Распределения P и Q")
    axes4[0].legend()
    axes4[0].grid(True, alpha=0.3)

    axes4[1].bar(
        ["KL(P‖Q)", "KL(Q‖P)"],
        [max(0, kl_pq), max(0, kl_qp)],
        color=["steelblue", "darkorange"],
        edgecolor="black",
    )
    axes4[1].set_ylabel("KL Divergence (nats)")
    axes4[1].set_title("Асимметрия KL-дивергенции")
    for i, v in enumerate([max(0, kl_pq), max(0, kl_qp)]):
        axes4[1].text(i, v + 0.01, f"{v:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes4[1].grid(True, axis="y", alpha=0.3)

    st.pyplot(fig4)

    col_a, col_b = st.columns(2)
    col_a.metric("KL(P ‖ Q)", f"{max(0, kl_pq):.4f} nats")
    col_b.metric("KL(Q ‖ P)", f"{max(0, kl_qp):.4f} nats")

    if abs(kl_pq - kl_qp) < 0.001:
        st.success("KL(P‖Q) ≈ KL(Q‖P) — распределения почти совпадают или симметричны.")
    else:
        st.info(f"KL асимметрична: разница между KL(P‖Q) и KL(Q‖P) = {abs(kl_pq - kl_qp):.4f}.")

    st.markdown("---")
    st.markdown(
        r"""
**KL в VAE:** ELBO = Реконструкционная потеря + KL-регуляризация

$$\mathcal{L}_\text{VAE} = \mathbb{E}_{q_\phi(z|x)}\left[\log p_\theta(x|z)\right] - D_\text{KL}\!\left(q_\phi(z|x) \| p(z)\right)$$

Для $q_\phi = \mathcal{N}(\mu, \sigma^2)$ и $p(z) = \mathcal{N}(0,1)$:

$$D_\text{KL} = -\frac{1}{2}\left(1 + \log\sigma^2 - \mu^2 - \sigma^2\right)$$
"""
    )

    mu_vae = st.slider("μ энкодера (VAE)", -3.0, 3.0, 0.5, 0.1)
    log_var_vae = st.slider("log σ² энкодера (VAE)", -3.0, 3.0, 0.0, 0.1)
    sigma2_vae = np.exp(log_var_vae)
    kl_vae = -0.5 * (1 + log_var_vae - mu_vae ** 2 - sigma2_vae)

    st.metric(f"KL(N({mu_vae:.1f}, {sigma2_vae:.2f}) ‖ N(0,1))", f"{kl_vae:.4f} nats")
    if kl_vae < 0.01:
        st.success("Латентное распределение близко к стандартному нормальному — хорошая регуляризация.")
    elif kl_vae > 5:
        st.warning("Большая KL — энкодер сильно отклоняется от prior N(0,1).")
