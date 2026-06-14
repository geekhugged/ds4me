import streamlit as st

st.set_page_config(page_title="DS4ME — Симуляции", page_icon="🧪", layout="wide")

home = st.Page("home.py", title="Главная", icon="🏠", default=True)
problem_setup = st.Page(
    "topics/problem_setup.py",
    title="1. Постановка задачи ML",
    icon="🧭",
)
linear_regression = st.Page(
    "topics/linear_regression.py",
    title="2. Линейная регрессия и МНК",
    icon="📈",
)
regression_metrics = st.Page(
    "topics/regression_metrics.py",
    title="3. Метрики качества регрессии",
    icon="📏",
)
bias_variance = st.Page(
    "topics/bias_variance.py",
    title="4. Bias-variance tradeoff",
    icon="⚖️",
)
logistic_regression = st.Page(
    "topics/logistic_regression.py",
    title="5. Логистическая регрессия",
    icon="🔀",
)
hypothesis_testing = st.Page(
    "topics/hypothesis_testing.py",
    title="6. Проверка статистических гипотез",
    icon="🧪",
)
confidence_intervals = st.Page(
    "topics/confidence_intervals.py",
    title="7. Доверительные интервалы",
    icon="📐",
)
ab_test_design = st.Page(
    "topics/ab_test_design.py",
    title="8. Дизайн A/B-теста и размер выборки",
    icon="🧮",
)
multiple_testing = st.Page(
    "topics/multiple_testing.py",
    title="9. Множественное тестирование (Bonferroni/FDR)",
    icon="🔁",
)
causal_dag = st.Page(
    "topics/causal_dag.py",
    title="10. DAG, конфаундеры и селекшн-bias",
    icon="🕸️",
)

pg = st.navigation(
    [
        home,
        problem_setup,
        linear_regression,
        regression_metrics,
        bias_variance,
        logistic_regression,
        hypothesis_testing,
        confidence_intervals,
        ab_test_design,
        multiple_testing,
        causal_dag,
    ]
)
pg.run()
