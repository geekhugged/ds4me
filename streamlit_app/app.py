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

pg = st.navigation(
    [home, problem_setup, linear_regression, regression_metrics, bias_variance, logistic_regression]
)
pg.run()
