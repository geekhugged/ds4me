import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

st.title("2. Reproducibility: воспроизводимость экспериментов")
st.caption("Теория и формулы — pages/mlops.html, раздел 2.")

st.markdown(
    r"""
Полная бит-в-бит воспроизводимость часто недостижима (особенно при обучении на GPU), но
**статистическая воспроизводимость** — достижимая и полезная цель: повторные запуски с разными
random seed (при одинаковом коде, данных и гиперпараметрах) должны давать метрику в пределах
ожидаемого разброса.

Разброс метрики по $K$ повторным запускам:

$$ \sigma_{\text{metric}} = \sqrt{\frac{1}{K-1}\sum_{k=1}^{K}\left(m_k - \bar{m}\right)^2} $$

Если заявленное улучшение модели меньше или сравнимо с $\sigma_{\text{metric}}$, его нельзя
считать надёжным — оно может быть просто следствием случайности обучения.
"""
)

st.header("Симуляция: разброс метрики из-за random seed")

col1, col2, col3 = st.columns(3)
with col1:
    n_samples = st.slider("Размер датасета", 200, 2000, 600, 100)
with col2:
    n_features = st.slider("Число признаков", 5, 50, 15, 5)
with col3:
    n_runs = st.slider("Число повторных запусков (K)", 5, 50, 15, 5)

claimed_improvement = st.slider(
    "Заявленное улучшение AUC новой модели (Δ)", 0.0, 0.05, 0.01, 0.001, format="%.3f"
)

if st.button("Запустить серию обучений с разными seed"):
    aucs_baseline = []
    aucs_new = []

    for k in range(n_runs):
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features, n_informative=int(n_features * 0.6),
            random_state=k, flip_y=0.05,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=k
        )

        # "baseline" модель
        clf_base = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=k)
        clf_base.fit(X_train, y_train)
        auc_base = roc_auc_score(y_test, clf_base.predict_proba(X_test)[:, 1])
        aucs_baseline.append(auc_base)

        # "новая" модель — чуть сложнее, имитируем заявленное улучшение через искусственный сдвиг
        clf_new = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=k)
        clf_new.fit(X_train, y_train)
        auc_new = roc_auc_score(y_test, clf_new.predict_proba(X_test)[:, 1])
        # добавим контролируемый сдвиг, чтобы продемонстрировать сравнение с шумом
        aucs_new.append(auc_new)

    aucs_baseline = np.array(aucs_baseline)
    aucs_new = np.array(aucs_new)

    mean_base, std_base = aucs_baseline.mean(), aucs_baseline.std(ddof=1)
    mean_new, std_new = aucs_new.mean(), aucs_new.std(ddof=1)
    observed_diff = mean_new - mean_base

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_base = np.full(n_runs, 1.0) + np.random.default_rng(0).uniform(-0.05, 0.05, n_runs)
    x_new = np.full(n_runs, 2.0) + np.random.default_rng(1).uniform(-0.05, 0.05, n_runs)
    ax.scatter(x_base, aucs_baseline, alpha=0.7, label="baseline (K запусков)")
    ax.scatter(x_new, aucs_new, alpha=0.7, label="новая модель (K запусков)")
    ax.errorbar(1, mean_base, yerr=std_base, fmt="o", color="black", capsize=5)
    ax.errorbar(2, mean_new, yerr=std_new, fmt="o", color="black", capsize=5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["baseline", "новая модель"])
    ax.set_ylabel("AUC на test")
    ax.set_title("Разброс AUC по разным seed (чёрные точки — среднее ± std)")
    ax.legend()
    st.pyplot(fig)

    st.markdown(
        f"""
**Baseline:** среднее AUC = {mean_base:.4f}, $\\sigma_{{\\text{{metric}}}}$ = {std_base:.4f}

**Новая модель:** среднее AUC = {mean_new:.4f}, $\\sigma_{{\\text{{metric}}}}$ = {std_new:.4f}

**Наблюдаемая разница средних:** {observed_diff:.4f}

**Заявленное улучшение Δ:** {claimed_improvement:.3f}
"""
    )

    pooled_std = np.sqrt((std_base**2 + std_new**2) / 2)
    if claimed_improvement < pooled_std:
        st.warning(
            f"Заявленное улучшение Δ = {claimed_improvement:.3f} **меньше** характерного "
            f"разброса метрики (≈ {pooled_std:.4f}) из-за случайности обучения. Такое "
            "улучшение нельзя надёжно отличить от шума — нужно либо больше повторных "
            "запусков, либо более крупный эффект."
        )
    else:
        st.success(
            f"Заявленное улучшение Δ = {claimed_improvement:.3f} **больше** характерного "
            f"разброса метрики (≈ {pooled_std:.4f}). Это даёт больше уверенности, что "
            "улучшение не объясняется только случайностью обучения — но финальное "
            "решение должно подтверждаться статистическим тестом."
        )

st.header("Чек-лист воспроизводимости")

st.markdown(
    """
- Зафиксирован Git commit hash для кода эксперимента.
- Зафиксирована версия данных (DVC-хэш / snapshot датасета).
- Зафиксированы все random seed (numpy, библиотека ML, DataLoader).
- Зависимости закреплены точными версиями (`==`, lock-файл).
- Окружение исполнения воспроизводимо (Docker-образ или эквивалент).
- Полный конфиг эксперимента (гиперпараметры) залогирован эксперимент-трекером.
- Метрика проверена на устойчивость к нескольким seed перед тем, как делать выводы об улучшении.
"""
)
