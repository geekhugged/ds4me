import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("14. Метод максимального правдоподобия (MLE)")
st.caption("Теория и формулы — pages/ml-theory.html, раздел 14.")

st.markdown(
    r"""
Log-likelihood для $m$ независимых наблюдений: $\ell(\theta) = \sum_{i=1}^m \log p(x_i \mid \theta)$.

MLE: $\hat{\theta}_{\text{MLE}} = \arg\max_\theta \ell(\theta)$.

Для нормального распределения: $\hat{\mu}_{\text{MLE}} = \frac{1}{m}\sum x_i$,
$\hat{\sigma}^2_{\text{MLE}} = \frac{1}{m}\sum (x_i - \hat{\mu})^2$.
"""
)

st.header("Подбор параметров нормального распределения по данным и likelihood surface")

true_mu = st.slider("Истинное среднее (для генерации данных)", -5.0, 5.0, 1.0, 0.1)
true_sigma = st.slider("Истинное стандартное отклонение (для генерации данных)", 0.3, 3.0, 1.5, 0.1)
m = st.slider("Размер выборки m", 5, 500, 50, 5)
seed = st.slider("Random seed", 0, 100, 0, 1)

rng = np.random.default_rng(seed)
data = rng.normal(loc=true_mu, scale=true_sigma, size=m)

mu_hat = data.mean()
sigma_hat = data.std(ddof=0)

st.write(
    f"MLE-оценки по сгенерированной выборке: "
    f"$\\hat\\mu_{{MLE}} = {mu_hat:.3f}$, $\\hat\\sigma_{{MLE}} = {sigma_hat:.3f}$ "
    f"(истинные значения: $\\mu={true_mu}$, $\\sigma={true_sigma}$)."
)

mu_grid = np.linspace(mu_hat - 3, mu_hat + 3, 150)
sigma_grid = np.linspace(max(0.1, sigma_hat - 2), sigma_hat + 2, 150)
MU, SIGMA = np.meshgrid(mu_grid, sigma_grid)

log_likelihood = np.zeros_like(MU)
for i in range(MU.shape[0]):
    for j in range(MU.shape[1]):
        mu_ij = MU[i, j]
        sigma_ij = SIGMA[i, j]
        ll = -0.5 * m * np.log(2 * np.pi * sigma_ij ** 2) - np.sum((data - mu_ij) ** 2) / (2 * sigma_ij ** 2)
        log_likelihood[i, j] = ll

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

axes[0].hist(data, bins=20, density=True, alpha=0.5, color="steelblue", label="данные")
x_plot = np.linspace(data.min() - 1, data.max() + 1, 300)
pdf_mle = (1.0 / (np.sqrt(2 * np.pi) * sigma_hat)) * np.exp(-((x_plot - mu_hat) ** 2) / (2 * sigma_hat ** 2))
pdf_true = (1.0 / (np.sqrt(2 * np.pi) * true_sigma)) * np.exp(-((x_plot - true_mu) ** 2) / (2 * true_sigma ** 2))
axes[0].plot(x_plot, pdf_mle, color="crimson", label="MLE-оценка плотности")
axes[0].plot(x_plot, pdf_true, color="darkgreen", linestyle="--", label="истинная плотность")
axes[0].set_title("Данные и подобранное нормальное распределение")
axes[0].set_xlabel("x")
axes[0].legend(fontsize=8)

cs = axes[1].contourf(MU, SIGMA, log_likelihood, levels=40, cmap="viridis")
axes[1].scatter([mu_hat], [sigma_hat], color="red", marker="*", s=200, label="MLE optimum", zorder=3)
axes[1].scatter([true_mu], [true_sigma], color="white", edgecolor="black", marker="o", s=80, label="истинные параметры", zorder=3)
axes[1].set_xlabel(r"$\mu$")
axes[1].set_ylabel(r"$\sigma$")
axes[1].set_title("Поверхность log-likelihood")
axes[1].legend(fontsize=8)
fig.colorbar(cs, ax=axes[1], label="log-likelihood")

st.pyplot(fig)

st.markdown(
    """
Красная звезда на поверхности log-likelihood — найденная MLE-оценка $(\\hat\\mu, \\hat\\sigma)$,
которая в точности совпадает с выборочным средним и выборочным стандартным отклонением.
Белая точка — истинные параметры, использованные для генерации данных. При увеличении
размера выборки $m$ MLE-оценка сходится к истинным параметрам (состоятельность оценки),
а поверхность log-likelihood становится более «острой» вокруг максимума (растёт точность
оценки, уменьшается её дисперсия).
"""
)
