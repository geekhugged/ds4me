import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.title("8. Регуляризация: L1 (Lasso), L2 (Ridge), ElasticNet")
st.caption("Теория и формулы — pages/ml-theory.html, раздел 8.")

st.markdown(
    r"""
Ridge: $J(\theta) = J_0(\theta) + \lambda \|\theta\|_2^2$ — сжимает коэффициенты к нулю,
но не зануляет их.

Lasso: $J(\theta) = J_0(\theta) + \lambda \|\theta\|_1$ — может занулять коэффициенты
(автоматический отбор признаков).

ElasticNet комбинирует оба штрафа с весом $\alpha$.
"""
)

st.header("Влияние силы регуляризации на коэффициенты и переобучение")

n_features = st.slider("Число признаков", 5, 30, 15, 1)
n_informative = st.slider("Из них информативных", 1, n_features, 4, 1)
n_samples = st.slider("Число объектов", 20, 200, 50, 5)
noise = st.slider("Шум", 0.0, 5.0, 1.0, 0.1)
method = st.radio("Метод регуляризации", ["Ridge (L2)", "Lasso (L1)", "ElasticNet"], horizontal=True)
log_lambda = st.slider("log10(λ)", -3.0, 2.0, -1.0, 0.1)
lam = 10 ** log_lambda

rng = np.random.default_rng(42)
X = rng.normal(size=(n_samples, n_features))
true_coef = np.zeros(n_features)
true_coef[:n_informative] = rng.uniform(2, 5, size=n_informative) * rng.choice([-1, 1], size=n_informative)
y = X @ true_coef + rng.normal(scale=noise, size=n_samples)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

if method == "Ridge (L2)":
    model = Ridge(alpha=lam)
elif method == "Lasso (L1)":
    model = Lasso(alpha=lam, max_iter=20000)
else:
    model = ElasticNet(alpha=lam, l1_ratio=0.5, max_iter=20000)

model.fit(X_train_s, y_train)
ols = LinearRegression().fit(X_train_s, y_train)

train_mse = mean_squared_error(y_train, model.predict(X_train_s))
test_mse = mean_squared_error(y_test, model.predict(X_test_s))
ols_train_mse = mean_squared_error(y_train, ols.predict(X_train_s))
ols_test_mse = mean_squared_error(y_test, ols.predict(X_test_s))

fig, ax = plt.subplots(figsize=(9, 4.5))
idx = np.arange(n_features)
width = 0.35
ax.bar(idx - width / 2, ols.coef_, width, label="OLS (без регуляризации)", alpha=0.7)
ax.bar(idx + width / 2, model.coef_, width, label=f"{method}, λ={lam:.3f}", alpha=0.7)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("индекс признака")
ax.set_ylabel("коэффициент")
ax.set_title("Коэффициенты модели (первые n_informative — реально значимые)")
ax.set_xticks(idx)
ax.legend()
st.pyplot(fig)

n_zero = np.sum(np.abs(model.coef_) < 1e-6)
col1, col2, col3 = st.columns(3)
col1.metric("Train MSE", f"{train_mse:.3f}", delta=f"{train_mse - ols_train_mse:+.3f} vs OLS")
col2.metric("Test MSE", f"{test_mse:.3f}", delta=f"{test_mse - ols_test_mse:+.3f} vs OLS", delta_color="inverse")
col3.metric("Коэффициентов ≈ 0", f"{n_zero}/{n_features}")

st.header("Путь регуляризации: коэффициенты vs λ")

lambdas = np.logspace(-3, 2, 40)
coef_paths = []
for l in lambdas:
    if method == "Ridge (L2)":
        m = Ridge(alpha=l)
    elif method == "Lasso (L1)":
        m = Lasso(alpha=l, max_iter=20000)
    else:
        m = ElasticNet(alpha=l, l1_ratio=0.5, max_iter=20000)
    m.fit(X_train_s, y_train)
    coef_paths.append(m.coef_)
coef_paths = np.array(coef_paths)

fig2, ax2 = plt.subplots(figsize=(8, 4.5))
for j in range(n_features):
    style = "-" if j < n_informative else "--"
    ax2.plot(lambdas, coef_paths[:, j], style, alpha=0.8 if j < n_informative else 0.4)
ax2.axvline(lam, color="red", linestyle=":", label=f"текущая λ={lam:.3f}")
ax2.set_xscale("log")
ax2.set_xlabel("λ (лог. шкала)")
ax2.set_ylabel("коэффициент")
ax2.set_title("Сплошные линии — информативные признаки, штрих — шумовые")
ax2.legend()
st.pyplot(fig2)

st.markdown(
    """
При увеличении λ все коэффициенты сжимаются к нулю. У Lasso шумовые признаки (штриховые линии)
часто обнуляются точно — это и есть встроенный отбор признаков. У Ridge коэффициенты лишь
уменьшаются, но почти никогда не становятся точно нулевыми.
"""
)
