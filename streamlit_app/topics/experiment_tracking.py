import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

st.title("4. Эксперимент-трекинг: MLflow и Weights & Biases")
st.caption("Теория и формулы — pages/mlops.html, раздел 4.")

st.markdown(
    r"""
Эксперимент-трекинг записывает для каждого **run**: параметры $\theta$ (гиперпараметры),
метрики $\text{metric}(\theta)$, артефакты и метаданные. По сути это сбор истории решения
задачи оптимизации

$$ \theta^{*} = \arg\max_{\theta \in \Theta} \; \mathbb{E}\left[ \text{metric}(\theta) \right] $$

Симуляция ниже запускает несколько "экспериментов" со случайными гиперпараметрами
(аналог random search), логирует каждый run в таблицу (как сделал бы MLflow/W&B) и показывает,
как выбрать лучший run.
"""
)

st.header("Симуляция: серия запусков (runs) с разными гиперпараметрами")

col1, col2, col3 = st.columns(3)
with col1:
    n_runs = st.slider("Число запусков (runs)", 3, 30, 10, 1)
with col2:
    n_samples = st.slider("Размер датасета", 300, 2000, 800, 100)
with col3:
    seed = st.slider("Random seed для генерации датасета", 0, 100, 42, 1)

st.markdown("Диапазоны гиперпараметров для случайного поиска:")
col4, col5 = st.columns(2)
with col4:
    n_estimators_range = st.slider("n_estimators (диапазон)", 10, 300, (20, 200), 10)
with col5:
    max_depth_range = st.slider("max_depth (диапазон)", 1, 20, (2, 12), 1)

if st.button("Запустить серию экспериментов (логирование run'ов)"):
    X, y = make_classification(
        n_samples=n_samples, n_features=20, n_informative=10, random_state=seed, flip_y=0.05
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    rng = np.random.default_rng(seed)
    runs = []
    for run_id in range(1, n_runs + 1):
        n_est = int(rng.integers(n_estimators_range[0], n_estimators_range[1] + 1))
        max_d = int(rng.integers(max_depth_range[0], max_depth_range[1] + 1))
        lr_seed = int(rng.integers(0, 10000))

        clf = RandomForestClassifier(
            n_estimators=n_est, max_depth=max_d, random_state=lr_seed
        )
        clf.fit(X_train, y_train)
        auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

        runs.append(
            {
                "run_id": run_id,
                "n_estimators": n_est,
                "max_depth": max_d,
                "random_state": lr_seed,
                "auc": round(auc, 4),
            }
        )

    df = pd.DataFrame(runs)
    best_run = df.loc[df["auc"].idxmax()]

    st.markdown("### Таблица run'ов (как UI MLflow Experiments)")
    st.dataframe(
        df.style.apply(
            lambda row: ["background-color: #d4edda" if row["run_id"] == best_run["run_id"] else "" for _ in row],
            axis=1,
        ),
        width="stretch",
    )

    st.markdown(
        f"""
**Лучший run: #{int(best_run['run_id'])}** с AUC = {best_run['auc']:.4f}
(`n_estimators={int(best_run['n_estimators'])}`, `max_depth={int(best_run['max_depth'])}`)

Именно этот run — кандидат на регистрацию как новая версия модели в Model Registry.
"""
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(df["n_estimators"], df["auc"], c=df["max_depth"], cmap="viridis", s=60)
    axes[0].scatter(best_run["n_estimators"], best_run["auc"], color="red", marker="*", s=250, label="лучший run")
    axes[0].set_xlabel("n_estimators")
    axes[0].set_ylabel("AUC")
    axes[0].set_title("AUC vs n_estimators (цвет = max_depth)")
    axes[0].legend()

    axes[1].bar(df["run_id"], df["auc"], color=["seagreen" if r == best_run["run_id"] else "steelblue" for r in df["run_id"]])
    axes[1].set_xlabel("run_id")
    axes[1].set_ylabel("AUC")
    axes[1].set_title("AUC по каждому run")

    st.pyplot(fig)

    st.markdown(
        """
В реальном MLflow/W&B каждый такой run также сохранил бы:
- версию кода (Git commit),
- версию данных (например, DVC-хэш),
- полный конфиг (все гиперпараметры, включая фиксированные),
- артефакты (сохранённую модель, графики важности признаков, confusion matrix).

Это позволяет позже воспроизвести лучший run точно так же, как он был получен.
"""
    )
