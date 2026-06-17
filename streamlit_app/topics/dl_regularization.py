import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("7. Регуляризация в DL: Dropout, BatchNorm/LayerNorm, Weight Decay, Early Stopping")
st.caption("Теория и формулы — pages/deep-learning.html, раздел 7.")

st.markdown(
    r"""
**Dropout**: на обучении зануляет нейрон с вероятностью $p$, масштабируя оставшиеся:
$$\tilde{a} = \frac{m \odot a}{1-p}, \quad m \sim \text{Bernoulli}(1-p)$$

**Batch/Layer Norm**: нормализуют активации (по батчу или по признакам), затем применяют
обучаемые $\gamma,\beta$:
$$\hat{x} = \frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}}, \qquad y = \gamma\hat{x}+\beta$$

**Weight decay**: штраф $\frac{\lambda}{2}\|W\|_2^2$ в функции потерь — сжимает веса к нулю.

**Early stopping**: остановка обучения при росте ошибки на валидации.
"""
)

st.header("1. Dropout: эффект на обучение простой сети (numpy)")

st.markdown(
    "Обучаем 2-слойную сеть (numpy, ручной backprop) на синтетических данных с шумом, "
    "сравнивая разные значения dropout rate $p$."
)

col1, col2, col3 = st.columns(3)
with col1:
    n_samples_do = st.slider("Число обучающих примеров", 20, 200, 40, 10)
    noise_do = st.slider("Уровень шума в данных", 0.0, 2.0, 0.8, 0.1)
with col2:
    hidden_do = st.slider("Размер скрытого слоя", 8, 128, 64, 8)
    epochs_do = st.slider("Число эпох", 50, 1000, 400, 50)
with col3:
    lr_do = st.slider("Learning rate", 0.001, 0.5, 0.05, 0.001)
    dropout_p = st.slider("Dropout rate p", 0.0, 0.8, 0.3, 0.05)

rng = np.random.default_rng(0)
X_train = np.sort(rng.uniform(-3, 3, size=(n_samples_do, 1)), axis=0)
y_train = np.sin(X_train) + rng.normal(scale=noise_do, size=(n_samples_do, 1)) * 0.3
X_test = np.linspace(-3, 3, 200).reshape(-1, 1)
y_test_true = np.sin(X_test)

def train_mlp_dropout(X, y, hidden, epochs, lr, p, seed=1):
    r = np.random.default_rng(seed)
    W1 = r.normal(0, np.sqrt(2 / 1), size=(1, hidden))
    b1 = np.zeros(hidden)
    W2 = r.normal(0, np.sqrt(2 / hidden), size=(hidden, 1))
    b2 = np.zeros(1)
    train_losses, test_losses = [], []
    clip = 5.0
    for ep in range(epochs):
        z1 = X @ W1 + b1
        a1 = np.maximum(0, z1)
        if p > 0:
            mask = r.binomial(1, 1 - p, size=a1.shape)
            a1_drop = a1 * mask / (1 - p)
        else:
            a1_drop = a1
        y_pred = a1_drop @ W2 + b2
        loss = np.mean((y_pred - y) ** 2)

        d_pred = 2 * (y_pred - y) / len(y)
        dW2 = np.clip(a1_drop.T @ d_pred, -clip, clip)
        db2 = np.clip(d_pred.sum(axis=0), -clip, clip)
        da1 = d_pred @ W2.T
        if p > 0:
            da1 = da1 * mask / (1 - p)
        dz1 = da1 * (z1 > 0)
        dW1 = np.clip(X.T @ dz1, -clip, clip)
        db1 = np.clip(dz1.sum(axis=0), -clip, clip)

        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2

        # eval (no dropout)
        z1e = X_test @ W1 + b1
        a1e = np.maximum(0, z1e)
        y_pred_test = a1e @ W2 + b2
        test_loss = np.mean((y_pred_test - y_test_true) ** 2)

        train_losses.append(loss)
        test_losses.append(test_loss)
    z1e = X_test @ W1 + b1
    a1e = np.maximum(0, z1e)
    final_pred = a1e @ W2 + b2
    return train_losses, test_losses, final_pred

train_losses, test_losses, final_pred = train_mlp_dropout(
    X_train, y_train, hidden_do, epochs_do, lr_do, dropout_p
)
train_losses_nd, test_losses_nd, final_pred_nd = train_mlp_dropout(
    X_train, y_train, hidden_do, epochs_do, lr_do, 0.0
)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].scatter(X_train, y_train, color="black", s=20, label="Train data", zorder=5)
axes[0].plot(X_test, y_test_true, color="gray", lw=1.5, ls="--", label="Истинная sin(x)")
axes[0].plot(X_test, final_pred_nd, color="firebrick", lw=2, label="Без dropout (p=0)")
axes[0].plot(X_test, final_pred, color="steelblue", lw=2, label=f"С dropout (p={dropout_p})")
axes[0].set_title("Аппроксимация sin(x) с шумом")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].plot(test_losses_nd, color="firebrick", lw=1.5, label="Test loss, p=0")
axes[1].plot(test_losses, color="steelblue", lw=1.5, label=f"Test loss, p={dropout_p}")
axes[1].plot(train_losses_nd, color="firebrick", lw=1, ls=":", alpha=0.6, label="Train loss, p=0")
axes[1].plot(train_losses, color="steelblue", lw=1, ls=":", alpha=0.6, label=f"Train loss, p={dropout_p}")
axes[1].set_yscale("log")
axes[1].set_xlabel("Эпоха")
axes[1].set_ylabel("MSE (log scale)")
axes[1].set_title("Кривые обучения")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)
st.pyplot(fig)

st.caption(
    f"Финальный test MSE без dropout: {test_losses_nd[-1]:.4f}, "
    f"с dropout (p={dropout_p}): {test_losses[-1]:.4f}. "
    "При большом скрытом слое и шумных данных dropout обычно снижает test loss "
    "(борется с переобучением), но слишком высокий p замедляет обучение."
)

st.header("2. Batch Normalization: распределение активаций")

st.markdown(
    "Имитируем эффект сдвига распределения входов (covariate shift) на активации слоя "
    "с BatchNorm и без него."
)

col_bn1, col_bn2 = st.columns(2)
with col_bn1:
    shift = st.slider("Сдвиг распределения входа (μ)", -5.0, 5.0, 3.0, 0.5)
with col_bn2:
    scale_bn = st.slider("Масштаб распределения входа (σ)", 0.1, 5.0, 2.0, 0.1)

rng2 = np.random.default_rng(7)
x_raw = rng2.normal(loc=shift, scale=scale_bn, size=2000)
gamma_bn, beta_bn = 1.0, 0.0
eps_bn = 1e-5
mu_bn = x_raw.mean()
var_bn = x_raw.var()
x_bn = gamma_bn * (x_raw - mu_bn) / np.sqrt(var_bn + eps_bn) + beta_bn

z_bn = np.tanh(x_bn)
z_raw = np.tanh(x_raw)

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4))
axes2[0].hist(x_raw, bins=50, alpha=0.6, label="Вход (raw)", color="firebrick", density=True)
axes2[0].hist(x_bn, bins=50, alpha=0.6, label="После BatchNorm", color="steelblue", density=True)
axes2[0].set_title("Распределение пре-активации до/после BatchNorm")
axes2[0].legend()
axes2[0].grid(True, alpha=0.3)

axes2[1].hist(z_raw, bins=50, alpha=0.6, label="tanh(raw)", color="firebrick", density=True)
axes2[1].hist(z_bn, bins=50, alpha=0.6, label="tanh(BatchNorm)", color="steelblue", density=True)
axes2[1].set_title("Активация tanh после до/после BatchNorm")
axes2[1].legend()
axes2[1].grid(True, alpha=0.3)
st.pyplot(fig2)

st.caption(
    "При большом сдвиге μ без BatchNorm вход в tanh попадает в зону насыщения "
    "(значения близко к ±1, производная ≈ 0). BatchNorm возвращает распределение "
    "к нулевому среднему и единичной дисперсии, избегая насыщения независимо от сдвига входа."
)

st.header("3. Weight Decay: усадка весов")

lam_wd = st.slider("λ (коэффициент weight decay)", 0.0, 1.0, 0.1, 0.01)
n_steps_wd = st.slider("Число шагов", 10, 300, 100, 10)
lr_wd = 0.1

w = np.array([3.0, -2.0, 1.5, 0.5])
ws = [w.copy()]
for _ in range(n_steps_wd):
    grad = 0.01 * np.sign(w)  # small "data" gradient
    w = w - lr_wd * grad - lr_wd * lam_wd * w
    ws.append(w.copy())
ws = np.array(ws)

fig3, ax3 = plt.subplots(figsize=(9, 4))
for i in range(ws.shape[1]):
    ax3.plot(ws[:, i], label=f"w[{i}]")
ax3.axhline(0, color="gray", lw=0.8)
ax3.set_xlabel("Шаг")
ax3.set_ylabel("Значение веса")
ax3.set_title(f"Усадка весов при weight decay λ={lam_wd}")
ax3.legend()
ax3.grid(True, alpha=0.3)
st.pyplot(fig3)

st.header("4. Early Stopping: точка остановки")

st.markdown("Типичная картина train/val loss с переобучением — найдите оптимальную точку остановки.")

ep_max = st.slider("Число эпох (симуляция)", 20, 300, 150, 10)
ep_range = np.arange(ep_max)
train_loss_es = 2.0 * np.exp(-ep_range / 40) + 0.05
val_loss_es = 2.0 * np.exp(-ep_range / 40) + 0.05 + 0.0015 * np.maximum(0, ep_range - 60) ** 1.3 / 50

best_epoch = np.argmin(val_loss_es)

fig4, ax4 = plt.subplots(figsize=(9, 4))
ax4.plot(ep_range, train_loss_es, color="steelblue", label="Train loss")
ax4.plot(ep_range, val_loss_es, color="firebrick", label="Validation loss")
ax4.axvline(best_epoch, color="green", ls="--", label=f"Early stop (эпоха {best_epoch})")
ax4.set_xlabel("Эпоха")
ax4.set_ylabel("Loss")
ax4.set_title("Early Stopping: минимум validation loss")
ax4.legend()
ax4.grid(True, alpha=0.3)
st.pyplot(fig4)

st.success(
    f"Оптимальная точка остановки — эпоха {best_epoch} (минимум validation loss = "
    f"{val_loss_es[best_epoch]:.4f}). После неё train loss продолжает падать, "
    "а validation loss растёт — признак переобучения."
)
