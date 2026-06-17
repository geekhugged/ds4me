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
classification_metrics = st.Page(
    "topics/classification_metrics.py",
    title="6. Метрики качества классификации",
    icon="🎯",
)
gradient_descent = st.Page(
    "topics/gradient_descent.py",
    title="7. Градиентный спуск",
    icon="🏔️",
)
regularization = st.Page(
    "topics/regularization.py",
    title="8. Регуляризация (L1/L2/ElasticNet)",
    icon="🪢",
)
cross_validation = st.Page(
    "topics/cross_validation.py",
    title="9. Кросс-валидация и гиперпараметры",
    icon="🧩",
)
decision_trees = st.Page(
    "topics/decision_trees.py",
    title="10. Деревья решений",
    icon="🌳",
)
ensembles = st.Page(
    "topics/ensembles.py",
    title="11. Ансамбли (Bagging, RF, Boosting)",
    icon="🌲",
)
svm = st.Page(
    "topics/svm.py",
    title="12. SVM и kernel trick",
    icon="➗",
)
naive_bayes = st.Page(
    "topics/naive_bayes.py",
    title="13. Наивный Байес и теорема Байеса",
    icon="🎲",
)
mle = st.Page(
    "topics/mle.py",
    title="14. Метод максимального правдоподобия",
    icon="📊",
)
clustering = st.Page(
    "topics/clustering.py",
    title="15. Кластеризация (k-means, DBSCAN)",
    icon="🧬",
)
multi_armed_bandits = st.Page(
    "topics/multi_armed_bandits.py",
    title="16. Многорукие бандиты",
    icon="🎰",
)
dp_north_star = st.Page(
    "topics/north_star_aarrr_okr.py",
    title="1. North Star Metric, AARRR и OKR",
    icon="⭐",
)
dp_funnel_analytics = st.Page(
    "topics/funnel_analytics.py",
    title="2. Продуктовая аналитика: события, воронки, сегментация",
    icon="🔻",
)
dp_ab_test_hypothesis = st.Page(
    "topics/ab_test_hypothesis.py",
    title="3. A/B-тестирование: дизайн и гипотезы",
    icon="🧪",
)
dp_significance_power = st.Page(
    "topics/significance_power_sample_size.py",
    title="4. Значимость, p-value, мощность и размер выборки",
    icon="📐",
)
dp_retention_cohort = st.Page(
    "topics/retention_cohort.py",
    title="5. Retention, churn и cohort-анализ",
    icon="📅",
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
dl_weight_init = st.Page(
    "topics/weight_initialization.py",
    title="6. Инициализация весов (Xavier/He)",
    icon="🎯",
)
dl_regularization = st.Page(
    "topics/dl_regularization.py",
    title="7. Регуляризация в DL",
    icon="🧯",
)
dl_vanishing_gradients = st.Page(
    "topics/vanishing_gradients_resnet.py",
    title="8. Vanishing/exploding gradients и ResNet",
    icon="🪜",
)
dl_cnn = st.Page(
    "topics/cnn_basics.py",
    title="9. CNN: свёртки, пулинг, рецептивное поле",
    icon="🖼️",
)
dl_rnn = st.Page(
    "topics/rnn_lstm_gru.py",
    title="10. RNN, LSTM, GRU",
    icon="🔁",
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
            classification_metrics,
            gradient_descent,
            regularization,
            cross_validation,
            decision_trees,
            ensembles,
            svm,
            naive_bayes,
            mle,
            clustering,
            multi_armed_bandits,
        ],
        "1.1 Deep Learning": [
            dl_mlp,
            dl_activations,
            dl_backprop,
            dl_losses,
            dl_optimizers,
            dl_weight_init,
            dl_regularization,
            dl_vanishing_gradients,
            dl_cnn,
            dl_rnn,
        ],
        "2. MLOps": [
            ml_lifecycle,
            reproducibility,
            data_model_versioning,
            experiment_tracking,
            containerization,
            git_github,
        ],
        "3. Data Product": [
            dp_north_star,
            dp_funnel_analytics,
            dp_ab_test_hypothesis,
            dp_significance_power,
            dp_retention_cohort,
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
