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
ml_lifecycle = st.Page(
    "topics/ml_lifecycle.py",
    title="1. Жизненный цикл ML-проекта",
    icon="🔄",
)
reproducibility = st.Page(
    "topics/reproducibility.py",
    title="2. Reproducibility",
    icon="♻️",
)
data_model_versioning = st.Page(
    "topics/data_model_versioning.py",
    title="3. Версионирование данных и моделей",
    icon="🗂️",
)
experiment_tracking = st.Page(
    "topics/experiment_tracking.py",
    title="4. Эксперимент-трекинг",
    icon="📊",
)
containerization = st.Page(
    "topics/containerization.py",
    title="5. Контейнеризация и Kubernetes",
    icon="🐳",
)
leetcode_two_sum = st.Page(
    "topics/leetcode_two_sum.py",
    title="Two Sum",
    icon="🔢",
)
leetcode_longest_substring = st.Page(
    "topics/leetcode_longest_substring.py",
    title="Longest Substring Without Repeating Characters",
    icon="🪟",
)
leetcode_valid_parentheses = st.Page(
    "topics/leetcode_valid_parentheses.py",
    title="Valid Parentheses",
    icon="🧱",
)
leetcode_number_of_islands = st.Page(
    "topics/leetcode_number_of_islands.py",
    title="Number of Islands",
    icon="🏝️",
)
leetcode_merge_intervals = st.Page(
    "topics/leetcode_merge_intervals.py",
    title="Merge Intervals",
    icon="📏",
)

pg = st.navigation(
    {
        "Главная": [home],
        "1. ML — общая теория": [
            problem_setup,
            linear_regression,
            regression_metrics,
            bias_variance,
            logistic_regression,
        ],
        "2. MLOps": [
            ml_lifecycle,
            reproducibility,
            data_model_versioning,
            experiment_tracking,
            containerization,
        ],
        "5. Дизайн экспериментов": [
            hypothesis_testing,
            confidence_intervals,
            ab_test_design,
            multiple_testing,
            causal_dag,
        ],
        "6. LeetCode": [
            leetcode_two_sum,
            leetcode_longest_substring,
            leetcode_valid_parentheses,
            leetcode_number_of_islands,
            leetcode_merge_intervals,
        ],
    }
)
pg.run()
