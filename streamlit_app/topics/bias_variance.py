import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

st.title("4. Bias-variance tradeoff и переобучение / недообучение")
st.caption("Теория и формулы — pages/ml-theory.html, раздел 4.")

st.markdown(
    r"""
Ошибка модели разлагается как
$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2 + \text{Variance} + \sigma^2$.

Слишком простая модель (низкая степень полинома) — высокое смещение (**bias**, недообучение).
Слишком сложная модель (высокая степень) — высокая дисперсия (**variance**, переобучение):
хорошо описывает train, но плохо обобщается на test.
"""
)

n_samples = st.slider("Число точек", 20, 200, 40, 5)
noise = st.slider("Шум", 0.0, 1.0, 0.2, 0.05)
degree = st.slider("Степень полинома модели", 1, 15, 1, 1)

rng = np.random.default_rng(3)
X = rng.uniform(0, 1, size=n_samples)
y_true_fn = lambda x: np.sin(2 * np.pi * x)
y = y_true_fn(X) + rng.normal(scale=noise, size=n_samples)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
model.fit(X_train.reshape(-1, 1), y_train)

x_plot = np.linspace(0, 1, 200)
y_plot = model.predict(x_plot.reshape(-1, 1))

train_mse = mean_squared_error(y_train, model.predict(X_train.reshape(-1, 1)))
test_mse = mean_squared_error(y_test, model.predict(X_test.reshape(-1, 1)))

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(X_train, y_train, label="train", alpha=0.7)
ax.scatter(X_test, y_test, label="test", alpha=0.7)
ax.plot(x_plot, y_true_fn(x_plot), color="black", linestyle="--", label="истинная функция sin(2πx)")
ax.plot(x_plot, y_plot, color="red", label=f"модель, степень {degree}")
ax.set_ylim(-2, 2)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"train MSE = {train_mse:.3f}, test MSE = {test_mse:.3f}")
ax.legend()
st.pyplot(fig)

st.header("Кривая ошибки train/test от степени полинома")

degrees = np.arange(1, 16)
train_errors = []
test_errors = []
for d in degrees:
    m = make_pipeline(PolynomialFeatures(degree=int(d)), LinearRegression())
    m.fit(X_train.reshape(-1, 1), y_train)
    train_errors.append(mean_squared_error(y_train, m.predict(X_train.reshape(-1, 1))))
    test_errors.append(mean_squared_error(y_test, m.predict(X_test.reshape(-1, 1))))

fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.plot(degrees, train_errors, marker="o", label="train MSE")
ax2.plot(degrees, test_errors, marker="o", label="test MSE")
ax2.axvline(degree, color="gray", linestyle="--", label="выбранная степень")
ax2.set_yscale("log")
ax2.set_xlabel("степень полинома")
ax2.set_ylabel("MSE (лог. шкала)")
ax2.legend()
st.pyplot(fig2)

st.markdown(
    """
При малой степени обе ошибки высоки (**недообучение**). С ростом степени train-ошибка падает
почти до нуля, а test-ошибка после некоторого момента начинает расти (**переобучение**).
Оптимальная сложность модели — где test-ошибка минимальна.
"""
)
