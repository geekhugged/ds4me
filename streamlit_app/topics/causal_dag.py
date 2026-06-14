import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression

st.title("5. От корреляции к причинности: DAG, конфаундеры, селекшн-bias")
st.caption("Теория и формулы — pages/experiment-design.html, раздел 5.")

st.markdown(
    r"""
Если $Z$ — конфаундер ($X \leftarrow Z \rightarrow Y$), наблюдаемая корреляция между $X$ и $Y$
включает не только причинный эффект $\beta_{XY}$, но и слагаемое, связанное с влиянием $Z$ на обе
переменные:

$$ \text{Cov}(X, Y) = \beta_{XY} \cdot \text{Var}(X) + \gamma_{ZX} \cdot \gamma_{ZY} \cdot \text{Var}(Z) $$

Контроль за $Z$ (например, через регрессию с $Z$ как ковариатой) устраняет это смещение и даёт
несмещённую оценку $\beta_{XY}$.

Ниже — симуляция данных с конфаундером: сравниваем "наивную" оценку эффекта $X \to Y$ (простую
регрессию $Y$ на $X$) и скорректированную оценку (регрессию $Y$ на $X$ и $Z$).
"""
)

st.header("Симуляция конфаундинга")

col1, col2, col3 = st.columns(3)
with col1:
    true_effect = st.slider("Истинный эффект X на Y (β_XY)", -2.0, 2.0, 0.5, 0.1)
with col2:
    gamma_zx = st.slider("Влияние Z на X (γ_ZX)", -2.0, 2.0, 1.0, 0.1)
with col3:
    gamma_zy = st.slider("Влияние Z на Y (γ_ZY)", -2.0, 2.0, 1.0, 0.1)

n_samples = st.slider("Размер выборки", 50, 5000, 500, 50)
noise_level = st.slider("Уровень шума", 0.1, 3.0, 1.0, 0.1)

rng = np.random.default_rng(0)

# Z - конфаундер
Z = rng.normal(0, 1, size=n_samples)
# X зависит от Z
X = gamma_zx * Z + rng.normal(0, noise_level, size=n_samples)
# Y зависит от X (причинно) и от Z
Y = true_effect * X + gamma_zy * Z + rng.normal(0, noise_level, size=n_samples)

# Наивная регрессия Y ~ X
naive_model = LinearRegression().fit(X.reshape(-1, 1), Y)
naive_beta = naive_model.coef_[0]

# Скорректированная регрессия Y ~ X + Z
adj_model = LinearRegression().fit(np.column_stack([X, Z]), Y)
adj_beta = adj_model.coef_[0]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Истинный эффект β_XY", f"{true_effect:.3f}")
with col2:
    st.metric("Наивная оценка (Y ~ X)", f"{naive_beta:.3f}", delta=f"{naive_beta - true_effect:+.3f}")
with col3:
    st.metric("Скорректированная оценка (Y ~ X + Z)", f"{adj_beta:.3f}", delta=f"{adj_beta - true_effect:+.3f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Наивный график
axes[0].scatter(X, Y, alpha=0.3, s=10)
x_line = np.linspace(X.min(), X.max(), 100)
axes[0].plot(x_line, naive_model.predict(x_line.reshape(-1, 1)), color="crimson",
             label=f"наивная регрессия (β = {naive_beta:.2f})")
axes[0].plot(x_line, true_effect * x_line + Y.mean() - true_effect * X.mean(), color="black",
             linestyle="--", label=f"истинный эффект (β = {true_effect:.2f})")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Y")
axes[0].set_title("Наивная регрессия Y ~ X (без учёта Z)")
axes[0].legend(fontsize=8)

# Окраска по Z
sc = axes[1].scatter(X, Y, c=Z, cmap="coolwarm", alpha=0.5, s=10)
axes[1].set_xlabel("X")
axes[1].set_ylabel("Y")
axes[1].set_title("Те же данные, цвет = значение конфаундера Z")
cbar = plt.colorbar(sc, ax=axes[1])
cbar.set_label("Z")

st.pyplot(fig)

st.markdown(
    f"""
**Интерпретация:**

- Если $\\gamma_{{ZX}}$ и $\\gamma_{{ZY}}$ имеют одинаковый знак, наивная оценка обычно
  смещена в сторону переоценки эффекта (по модулю); если знаки разные — может даже изменить
  знак эффекта (как при парадоксе Симпсона).
- Контроль за $Z$ (правая регрессия с двумя переменными) приближает оценку к истинному
  значению $\\beta_{{XY}} = {true_effect}$.
- На правом графике видно, что цвет (значение $Z$) систематически связан и с $X$, и с $Y$ —
  именно эта связь создаёт "лишнюю" корреляцию между $X$ и $Y$ в наивной модели.
"""
)

st.header("Коллайдер: контроль за общим эффектом создаёт ложную ассоциацию")

st.markdown(
    r"""
Если $C$ — коллайдер ($X \rightarrow C \leftarrow Y$), то $X$ и $Y$ могут быть полностью
независимы в общей популяции, но при условии на $C$ (например, при фильтрации данных по
значению $C$) между ними возникает ложная ассоциация (Berkson's paradox / selection bias).
"""
)

n_collider = st.slider("Размер выборки для демонстрации коллайдера", 200, 5000, 1000, 100, key="collider_n")
collider_threshold = st.slider("Порог фильтрации по C (квантиль)", 0.1, 0.9, 0.5, 0.05)

rng2 = np.random.default_rng(1)
X_c = rng2.normal(0, 1, size=n_collider)
Y_c = rng2.normal(0, 1, size=n_collider)  # независимо от X
C = X_c + Y_c + rng2.normal(0, 0.5, size=n_collider)  # коллайдер: зависит от X и Y

threshold_value = np.quantile(C, collider_threshold)
selected = C >= threshold_value

corr_full = np.corrcoef(X_c, Y_c)[0, 1]
corr_selected = np.corrcoef(X_c[selected], Y_c[selected])[0, 1]

col1, col2 = st.columns(2)
with col1:
    st.metric("Корреляция X и Y во всей выборке", f"{corr_full:.3f}")
with col2:
    st.metric("Корреляция X и Y среди отфильтрованных по C", f"{corr_selected:.3f}")

fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.5))
axes2[0].scatter(X_c, Y_c, alpha=0.3, s=10, color="gray")
axes2[0].set_title(f"Вся выборка: corr(X, Y) = {corr_full:.3f}")
axes2[0].set_xlabel("X")
axes2[0].set_ylabel("Y")

axes2[1].scatter(X_c[~selected], Y_c[~selected], alpha=0.2, s=10, color="lightgray", label="не выбрано")
axes2[1].scatter(X_c[selected], Y_c[selected], alpha=0.4, s=10, color="crimson", label=f"C ≥ {threshold_value:.2f}")
axes2[1].set_title(f"После фильтрации по C: corr(X, Y) = {corr_selected:.3f}")
axes2[1].set_xlabel("X")
axes2[1].set_ylabel("Y")
axes2[1].legend(fontsize=8)

st.pyplot(fig2)

st.markdown(
    """
В исходной популяции X и Y независимы (корреляция близка к 0). Но если мы анализируем только
подвыборку с высоким значением коллайдера C (например, "пользователи, прошедшие модерацию"
или "пациенты, попавшие в больницу"), между X и Y возникает отрицательная корреляция —
чисто артефакт отбора, а не причинная связь.
"""
)
