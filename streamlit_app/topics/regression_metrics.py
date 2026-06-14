import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.title("3. Метрики качества для регрессии (MSE, MAE, RMSE, R²)")
st.caption("Теория и формулы — pages/ml-theory.html, раздел 3.")

st.markdown(
    r"""
$\text{MSE} = \frac{1}{m}\sum (y_i - \hat{y}_i)^2$,
$\text{MAE} = \frac{1}{m}\sum |y_i - \hat{y}_i|$,
$\text{RMSE} = \sqrt{\text{MSE}}$,
$R^2 = 1 - \dfrac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$.

MAE устойчива к выбросам, MSE/RMSE сильно штрафуют большие ошибки, $R^2$ показывает
долю объяснённой дисперсии (1 — идеально, 0 — как предсказание средним).
"""
)

st.header("Как выбросы влияют на метрики")

n_points = st.slider("Число точек", 10, 200, 50, 5)
noise = st.slider("Шум (стандартное отклонение)", 0.0, 5.0, 1.0, 0.1)
n_outliers = st.slider("Число выбросов", 0, 10, 0, 1)
outlier_magnitude = st.slider("Размер выброса (во сколько раз больше шума)", 1, 20, 10, 1)

rng = np.random.default_rng(7)
X = np.sort(rng.uniform(0, 10, size=n_points))
y = 2.0 + 1.5 * X + rng.normal(scale=noise, size=n_points)

outlier_idx = np.array([], dtype=int)
if n_outliers > 0:
    outlier_idx = rng.choice(n_points, size=min(n_outliers, n_points), replace=False)
    y_outliers = y.copy()
    y_outliers[outlier_idx] += rng.choice([-1, 1], size=len(outlier_idx)) * outlier_magnitude * (noise + 0.5)
else:
    y_outliers = y.copy()

model = LinearRegression().fit(X.reshape(-1, 1), y_outliers)
y_pred = model.predict(X.reshape(-1, 1))

mse = mean_squared_error(y_outliers, y_pred)
mae = mean_absolute_error(y_outliers, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_outliers, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))
mask = np.ones(n_points, dtype=bool)
mask[outlier_idx] = False
ax.scatter(X[mask], y_outliers[mask], label="обычные точки")
if len(outlier_idx) > 0:
    ax.scatter(X[outlier_idx], y_outliers[outlier_idx], color="red", label="выбросы", zorder=3)
ax.plot(X, y_pred, color="green", label="регрессия (МНК)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
st.pyplot(fig)

col1, col2, col3, col4 = st.columns(4)
col1.metric("MSE", f"{mse:.3f}")
col2.metric("MAE", f"{mae:.3f}")
col3.metric("RMSE", f"{rmse:.3f}")
col4.metric("R²", f"{r2:.3f}")

st.markdown(
    """
Увеличьте число и размер выбросов: значения **MSE** и **RMSE** будут расти заметно быстрее, чем
**MAE**, потому что ошибка возводится в квадрат. Линия регрессии при этом "тянется" к выбросам,
а $R^2$ может заметно упасть даже от одной аномальной точки.
"""
)
