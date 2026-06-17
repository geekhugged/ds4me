import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("13. Наивный байесовский классификатор и теорема Байеса")
st.caption("Теория и формулы — pages/ml-theory.html, раздел 13.")

st.markdown(
    r"""
Теорема Байеса:

$$ P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)} $$

где $P(A)$ — prior, $P(B\mid A)$ — likelihood, $P(A\mid B)$ — posterior.
Правило классификации наивного Байеса: $\hat{y} = \arg\max_y P(y)\prod_j P(x_j\mid y)$.
"""
)

st.header("Интерактивная теорема Байеса: медицинский тест")

st.markdown(
    """
Классический пример: тест на редкое заболевание. Подвигайте prior (распространённость
болезни в популяции) и качество теста (likelihood: чувствительность и специфичность),
чтобы увидеть, как меняется posterior — вероятность болезни при положительном тесте.
"""
)

prior = st.slider("Prior P(болезнь) — распространённость в популяции", 0.001, 0.5, 0.01, 0.001)
sensitivity = st.slider("Чувствительность теста P(тест+ | болезнь)", 0.5, 0.999, 0.95, 0.001)
specificity = st.slider("Специфичность теста P(тест− | нет болезни)", 0.5, 0.999, 0.95, 0.001)

p_disease = prior
p_no_disease = 1 - prior
p_pos_given_disease = sensitivity
p_pos_given_no_disease = 1 - specificity

p_pos = p_pos_given_disease * p_disease + p_pos_given_no_disease * p_no_disease
posterior = (p_pos_given_disease * p_disease) / p_pos if p_pos > 0 else 0.0

col1, col2, col3 = st.columns(3)
col1.metric("Prior P(болезнь)", f"{prior:.3%}")
col2.metric("P(тест положительный)", f"{p_pos:.3%}")
col3.metric("Posterior P(болезнь | тест+)", f"{posterior:.3%}")

fig, ax = plt.subplots(figsize=(7, 4.5))
labels = ["Prior\nP(болезнь)", "Posterior\nP(болезнь | тест+)"]
values = [prior, posterior]
bars = ax.bar(labels, values, color=["steelblue", "crimson"])
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3%}", ha="center", fontsize=10)
ax.set_ylim(0, max(values) * 1.3 + 0.05)
ax.set_ylabel("Вероятность")
ax.set_title("Обновление вероятности болезни после положительного теста")
st.pyplot(fig)

st.markdown(
    """
Даже при очень хорошем тесте (высокая чувствительность и специфичность) при низком
prior (редкая болезнь) posterior может оставаться далеко от 100% — большинство
положительных результатов в популяции с низкой распространённостью оказываются ложно
положительными. Это классическая иллюстрация важности учёта prior при интерпретации
результатов теста ("base rate fallacy").
"""
)

st.header("Наивный Байес как классификатор: 1D пример с двумя гауссовыми классами")

mu0 = st.slider("Среднее признака в классе 0", -5.0, 5.0, -1.5, 0.1)
mu1 = st.slider("Среднее признака в классе 1", -5.0, 5.0, 1.5, 0.1)
sigma = st.slider("Стандартное отклонение признака (одинаковое для обоих классов)", 0.2, 3.0, 1.0, 0.1)
class_prior1 = st.slider("Prior класса 1, P(y=1)", 0.05, 0.95, 0.5, 0.05)

x_grid = np.linspace(min(mu0, mu1) - 4 * sigma, max(mu0, mu1) + 4 * sigma, 500)


def gaussian_pdf(x, mu, sig):
    return (1.0 / (np.sqrt(2 * np.pi) * sig)) * np.exp(-((x - mu) ** 2) / (2 * sig ** 2))


p_x_given_0 = gaussian_pdf(x_grid, mu0, sigma)
p_x_given_1 = gaussian_pdf(x_grid, mu1, sigma)
prior0 = 1 - class_prior1
prior1 = class_prior1

joint0 = p_x_given_0 * prior0
joint1 = p_x_given_1 * prior1
posterior1 = joint1 / (joint0 + joint1 + 1e-12)

fig2, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(x_grid, p_x_given_0, label="P(x | y=0)", color="steelblue")
axes[0].plot(x_grid, p_x_given_1, label="P(x | y=1)", color="crimson")
axes[0].set_title("Правдоподобия (likelihood) классов")
axes[0].set_xlabel("x")
axes[0].set_ylabel("плотность")
axes[0].legend()

axes[1].plot(x_grid, posterior1, color="darkgreen")
axes[1].axhline(0.5, linestyle="--", color="gray")
axes[1].set_title("Апостериорная вероятность P(y=1 | x)")
axes[1].set_xlabel("x")
axes[1].set_ylabel("posterior")
st.pyplot(fig2)

st.markdown(
    """
Решающее правило MAP выбирает класс 1, если $P(y=1\\mid x) > 0.5$ (зелёная линия выше
серой штриховой). При увеличении prior класса 1 граница принятия решения смещается в
сторону класса 0 (модель более охотно предсказывает класс 1, даже при не очень
характерных значениях признака).
"""
)
