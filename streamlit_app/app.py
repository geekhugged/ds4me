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
dl_mlp = st.Page(
    "topics/mlp.py",
    title="1. Перцептрон и MLP",
    icon="🧠",
)
dl_activations = st.Page(
    "topics/activation_functions.py",
    title="2. Функции активации",
    icon="⚡",
)
dl_backprop = st.Page(
    "topics/backpropagation.py",
    title="3. Backpropagation",
    icon="🔄",
)
dl_losses = st.Page(
    "topics/loss_functions.py",
    title="4. Функции потерь",
    icon="📉",
)
dl_optimizers = st.Page(
    "topics/optimizers.py",
    title="5. Оптимизаторы",
    icon="🚀",
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
git_github = st.Page(
    "topics/git_github.py",
    title="6. Git и GitHub",
    icon="🔧",
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
mlsd_video_recsys = st.Page(
    "topics/mlsd_video_recsys.py",
    title="1. Видео-рекомендации (YouTube-style)",
    icon="📺",
)
mlsd_ad_ctr = st.Page(
    "topics/mlsd_ad_ctr.py",
    title="2. Предсказание CTR для рекламы",
    icon="📢",
)
mlsd_feed_ranking = st.Page(
    "topics/mlsd_feed_ranking.py",
    title="3. Ранжирование ленты соцсети",
    icon="📰",
)
mlsd_fraud_detection = st.Page(
    "topics/mlsd_fraud_detection.py",
    title="4. Детекция фрода в транзакциях",
    icon="🛡️",
)
mlsd_visual_search = st.Page(
    "topics/mlsd_visual_search.py",
    title="5. Визуальный поиск похожих товаров",
    icon="🔍",
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
        "1.1 Deep Learning": [
            dl_mlp,
            dl_activations,
            dl_backprop,
            dl_losses,
            dl_optimizers,
        ],
        "2. MLOps": [
            ml_lifecycle,
            reproducibility,
            data_model_versioning,
            experiment_tracking,
            containerization,
            git_github,
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
        "7. ML System Design": [
            mlsd_video_recsys,
            mlsd_ad_ctr,
            mlsd_feed_ranking,
            mlsd_fraud_detection,
            mlsd_visual_search,
        ],
    }
)
pg.run()
