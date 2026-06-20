import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

st.title("18. Feature engineering и работа с категориальными признаками")
st.caption("Теория и формулы — pages/ml-theory.html, раздел 18.")

st.markdown(
    r"""
Масштабирование числовых признаков:

- **StandardScaler**: $x' = \dfrac{x - \mu}{\sigma}$ — нулевое среднее, единичная дисперсия.
- **MinMaxScaler**: $x' = \dfrac{x - x_{\min}}{x_{\max} - x_{\min}}$ — сжатие в $[0, 1]$.
- **RobustScaler**: $x' = \dfrac{x - \text{med}(x)}{\text{IQR}(x)}$ — устойчив к выбросам.

Кодирование категорий: **one-hot** превращает признак с $k$ уровнями в $k$ бинарных
столбцов; **target encoding** заменяет категорию средним таргета по ней (со сглаживанием):

$$ \text{enc}(c) = \frac{n_c\,\bar{y}_c + m\,\bar{y}}{n_c + m}, $$

где $\bar{y}_c$ — среднее таргета в категории $c$, $n_c$ — её размер, $\bar{y}$ — глобальное
среднее, $m$ — параметр сглаживания (регуляризация к глобальному среднему).
"""
)

st.header("Эффект масштабирования на распределение")

scaler_name = st.radio(
    "Метод масштабирования", ["StandardScaler", "MinMaxScaler", "RobustScaler"], horizontal=True
)
outlier_frac = st.slider("Доля выбросов в данных", 0.0, 0.2, 0.05, 0.01)
n = st.slider("Размер выборки", 100, 1000, 400, 50)

rng = np.random.default_rng(0)
base = rng.normal(loc=50, scale=10, size=n)
n_out = int(outlier_frac * n)
if n_out > 0:
    base[:n_out] = rng.normal(loc=300, scale=30, size=n_out)
x = base.reshape(-1, 1)

scalers = {"StandardScaler": StandardScaler, "MinMaxScaler": MinMaxScaler, "RobustScaler": RobustScaler}
x_scaled = scalers[scaler_name]().fit_transform(x).ravel()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].hist(x.ravel(), bins=30, color="steelblue", alpha=0.8)
axes[0].set_title("До масштабирования")
axes[0].set_xlabel("значение признака")
axes[1].hist(x_scaled, bins=30, color="seagreen", alpha=0.8)
axes[1].set_title(f"После {scaler_name}")
axes[1].set_xlabel("масштабированное значение")
st.pyplot(fig)

st.write(
    f"После {scaler_name}: среднее = **{x_scaled.mean():.3f}**, "
    f"std = **{x_scaled.std():.3f}**, диапазон = **[{x_scaled.min():.2f}, {x_scaled.max():.2f}]**."
)

st.header("Кодирование категорий и качество модели")

st.markdown(
    "Создаём категориальный признак с высокой кардинальностью, связанный с таргетом, "
    "и сравниваем one-hot и target encoding на простой логистической регрессии."
)

k_levels = st.slider("Число уровней категории (кардинальность)", 3, 50, 20, 1)
signal = st.slider("Сила связи категории с таргетом", 0.0, 3.0, 1.5, 0.1)
smoothing = st.slider("Сглаживание target encoding (m)", 0.0, 50.0, 10.0, 1.0)

n_obs = 500
rng2 = np.random.default_rng(1)
cats = rng2.integers(0, k_levels, size=n_obs)
cat_effect = rng2.normal(scale=signal, size=k_levels)
num_feat = rng2.normal(size=n_obs)
logits = cat_effect[cats] + 0.5 * num_feat
proba = 1 / (1 + np.exp(-logits))
y = (rng2.random(n_obs) < proba).astype(int)

df = pd.DataFrame({"cat": cats, "num": num_feat})

# one-hot
X_ohe = pd.get_dummies(df["cat"].astype("category"), prefix="cat").to_numpy(dtype=float)
X_ohe = np.column_stack([X_ohe, df["num"].to_numpy()])

# target encoding со сглаживанием (на всём датасете для демонстрации)
global_mean = y.mean()
enc_map = {}
for c in range(k_levels):
    mask = cats == c
    n_c = mask.sum()
    mean_c = y[mask].mean() if n_c > 0 else global_mean
    enc_map[c] = (n_c * mean_c + smoothing * global_mean) / (n_c + smoothing)
X_te = np.column_stack([np.array([enc_map[c] for c in cats]), df["num"].to_numpy()])


@st.cache_data
def cv_score(X, y):
    model = LogisticRegression(max_iter=1000)
    return cross_val_score(model, X, y, cv=5, scoring="roc_auc").mean()


auc_ohe = cv_score(X_ohe, y)
auc_te = cv_score(X_te, y)

col1, col2, col3 = st.columns(3)
col1.metric("One-hot признаков", f"{X_ohe.shape[1]}")
col2.metric("ROC-AUC (one-hot)", f"{auc_ohe:.3f}")
col3.metric("ROC-AUC (target enc, 1 столбец)", f"{auc_te:.3f}")

st.markdown(
    """
При большой кардинальности one-hot раздувает размерность (много столбцов, разреженность,
риск переобучения на редких уровнях). Target encoding сжимает признак до одного числа и
часто не уступает по качеству, но требует аккуратности: без сглаживания и без расчёта на
out-of-fold данных он провоцирует утечку таргета (target leakage) и переоценивает качество.
Сглаживание (параметр $m$) подтягивает редкие категории к глобальному среднему и снижает
переобучение.
"""
)
