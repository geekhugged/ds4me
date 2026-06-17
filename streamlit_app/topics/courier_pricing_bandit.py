import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.title("16. Бандиты: кейс — динамическая надбавка курьеру в доставке")
st.caption(
    "Теория и формулы — pages/ml-theory.html, раздел 16, подраздел "
    "«Практический кейс: динамическая надбавка курьеру в доставке»."
)

st.markdown(
    r"""
**Постановка задачи.** Платформа доставки платит курьеру базовую цену `base_price` за заказ.
Готовая ML-модель (вне рамок этой темы) оценивает сложность заказа скором $s \in [0, 100]$.
Платформа хочет добавить надбавку $p$ сверху базовой цены, чтобы повысить шанс, что курьер
согласится на заказ — но не знает истинной чувствительности курьеров к надбавке. Это
**контекстный бандит**: контекст — бакет сложности заказа, руки — кандидатные уровни надбавки
из фиксированной сетки, награда — реализованная прибыль платформы по заказу.

**Модель принятия заказа.** Истинная (скрытая от алгоритма) надбавка, дающая 50% вероятность
принятия для сложности $s$:

$$ r(s) = r_{\min} + (r_{\max} - r_{\min}) \cdot \left(\frac{s}{100}\right)^{\gamma} $$

где $r_{\min}$, $r_{\max}$ — границы диапазона надбавки, а $\gamma$ — показатель,
контролирующий выпуклость роста требуемой компенсации с ростом сложности ($\gamma > 1$ означает,
что сложные заказы требуют непропорционально больше денег).

Вероятность принятия заказа курьером при выбранной надбавке $p$ и сложности $s$:

$$ P(\text{accept} \mid p, s) = \sigma\!\left(\frac{p - r(s)}{\tau}\right) = \frac{1}{1 + e^{-(p - r(s))/\tau}} $$

где $\tau$ (temperature) — резкость перехода между «почти всегда отказ» и «почти всегда согласие».

**Награда за раунд:**

$$ \text{reward} = \begin{cases} \text{value\_per\_order} - \text{base\_price} - p, & \text{заказ принят} \\ -\text{reject\_penalty}, & \text{заказ отклонён} \end{cases} $$

**Истинная ожидаемая награда** (известна только симуляции, используется лишь для оракула и
регрета, никогда не передаётся алгоритмам):

$$ \mathbb{E}[\text{reward} \mid p, s] = P(\text{accept})\cdot(\text{value\_per\_order} - \text{base\_price} - p) - (1 - P(\text{accept})) \cdot \text{reject\_penalty} $$

Оракульная рука для бакета — та надбавка из сетки, что максимизирует это выражение. Регрет на
раунде — разница между ожидаемой наградой оракульной руки и ожидаемой наградой выбранной руки
(по истинным вероятностям, а не по зашумлённому исходу), накопленная по раундам, отдельно для
каждой стратегии.

**Алгоритмы.** Каждая стратегия держит независимый бандит для каждого бакета сложности
(стратификация контекста):
1. **Epsilon-greedy** — на среднем накопленном вознаграждении по (бакет, рука).
2. **UCB1** (адаптация для не-бернуллиевских наград) — бонус
   $\sqrt{2\ln t / n}$, масштабированный диапазоном награды.
3. **Thompson Sampling (Гауссовский, Normal-Normal)** — апостериор по среднему вознаграждению
   каждой (бакет, рука) при известной дисперсии наблюдений.
"""
)

st.header("Параметры среды")

col1, col2, col3 = st.columns(3)
with col1:
    n_buckets = st.slider("Число бакетов сложности N_BUCKETS", 3, 8, 5, 1)
    n_arms = st.slider("Число уровней надбавки K (сетка)", 3, 12, 7, 1)
    n_rounds = st.slider("Число раундов T (заказов)", 200, 5000, 1500, 100)
with col2:
    r_min = st.slider("r_min (мин. требуемая надбавка)", 0, 200, 0, 10)
    r_max = st.slider("r_max (макс. требуемая надбавка)", 50, 500, 300, 10)
    gamma = st.slider("gamma (выпуклость роста требуемой надбавки)", 0.5, 3.0, 1.5, 0.1)
with col3:
    temperature = st.slider("temperature τ (резкость перехода accept/reject)", 1.0, 100.0, 20.0, 1.0)
    epsilon = st.slider("epsilon (для epsilon-greedy)", 0.0, 1.0, 0.1, 0.01)
    seed = st.slider("Random seed", 0, 100, 42, 1)

col4, col5, col6 = st.columns(3)
with col4:
    value_per_order = st.slider("value_per_order (доход платформы с заказа)", 100, 1000, 400, 10)
with col5:
    base_price = st.slider("base_price (базовая цена курьеру)", 0, 500, 150, 10)
with col6:
    reject_penalty = st.slider("reject_penalty (штраф за отказ курьера)", 0, 300, 50, 10)

sigma_obs = st.slider(
    "sigma_obs (известная std шума наблюдаемой награды, для Thompson Sampling)",
    1.0, 200.0, 60.0, 1.0,
)

st.markdown(
    "Сетка кандидатных уровней надбавки — равномерно от 0 до `r_max * 1.3`, "
    f"{n_arms} уровней."
)

# --- Environment setup ---
arm_grid = np.linspace(0, r_max * 1.3, n_arms)
bucket_edges = np.linspace(0, 100, n_buckets + 1)
bucket_midpoints = (bucket_edges[:-1] + bucket_edges[1:]) / 2


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def required_surcharge(s):
    return r_min + (r_max - r_min) * (s / 100.0) ** gamma


def accept_prob(p, s):
    return sigmoid((p - required_surcharge(s)) / temperature)


def expected_reward(p, s):
    pa = accept_prob(p, s)
    return pa * (value_per_order - base_price - p) - (1 - pa) * reject_penalty


# True per-bucket per-arm expected reward (oracle benchmark only)
true_er = np.array(
    [[expected_reward(p, bucket_midpoints[b]) for p in arm_grid] for b in range(n_buckets)]
)
oracle_arm_idx = np.argmax(true_er, axis=1)
oracle_er = true_er[np.arange(n_buckets), oracle_arm_idx]

st.write("Истинная требуемая надбавка (50% accept) по бакетам сложности (агенту неизвестна):")
bucket_labels = [f"[{bucket_edges[i]:.0f}, {bucket_edges[i+1]:.0f})" for i in range(n_buckets)]
st.write(
    {
        bucket_labels[b]: {
            "r(s) для медианной сложности": round(float(required_surcharge(bucket_midpoints[b])), 1),
            "оракульная надбавка": round(float(arm_grid[oracle_arm_idx[b]]), 1),
            "ожидаемая прибыль оракула": round(float(oracle_er[b]), 1),
        }
        for b in range(n_buckets)
    }
)


def bucket_of(s):
    idx = np.searchsorted(bucket_edges, s, side="right") - 1
    return int(np.clip(idx, 0, n_buckets - 1))


def simulate_orders(rng_orders, n_rounds):
    return rng_orders.uniform(0, 100, size=n_rounds)


def run_epsilon_greedy(complexities, rng_alg, rng_outcome):
    counts = np.zeros((n_buckets, n_arms))
    sums = np.zeros((n_buckets, n_arms))
    regrets = np.zeros(n_rounds)
    realized = np.zeros(n_rounds)
    accepted = np.zeros(n_rounds, dtype=bool)
    paid = np.zeros(n_rounds)
    cum_regret = 0.0

    for t in range(1, n_rounds + 1):
        s = complexities[t - 1]
        b = bucket_of(s)
        counts_b = counts[b]
        if rng_alg.random() < epsilon or np.any(counts_b == 0):
            if np.any(counts_b == 0):
                arm = int(np.argmin(counts_b))
            else:
                arm = rng_alg.integers(0, n_arms)
        else:
            means = sums[b] / np.maximum(counts_b, 1)
            arm = int(np.argmax(means))

        p = arm_grid[arm]
        pa = accept_prob(p, s)
        accept = rng_outcome.random() < pa
        reward = (value_per_order - base_price - p) if accept else -reject_penalty

        counts[b, arm] += 1
        sums[b, arm] += reward

        cum_regret += oracle_er[b] - true_er[b, arm]
        regrets[t - 1] = cum_regret
        realized[t - 1] = reward
        accepted[t - 1] = accept
        paid[t - 1] = p

    final_arm = np.array([int(np.argmax(counts[b])) for b in range(n_buckets)])
    return regrets, realized, accepted, paid, final_arm


def run_ucb1(complexities, rng_alg, rng_outcome):
    counts = np.zeros((n_buckets, n_arms))
    sums = np.zeros((n_buckets, n_arms))
    regrets = np.zeros(n_rounds)
    realized = np.zeros(n_rounds)
    accepted = np.zeros(n_rounds, dtype=bool)
    paid = np.zeros(n_rounds)
    cum_regret = 0.0
    reward_range = value_per_order + reject_penalty

    for t in range(1, n_rounds + 1):
        s = complexities[t - 1]
        b = bucket_of(s)
        counts_b = counts[b]
        if np.any(counts_b == 0):
            arm = int(np.argmin(counts_b))
        else:
            means = sums[b] / counts_b
            bonus = reward_range * np.sqrt(2 * np.log(t) / counts_b)
            arm = int(np.argmax(means + bonus))

        p = arm_grid[arm]
        pa = accept_prob(p, s)
        accept = rng_outcome.random() < pa
        reward = (value_per_order - base_price - p) if accept else -reject_penalty

        counts[b, arm] += 1
        sums[b, arm] += reward

        cum_regret += oracle_er[b] - true_er[b, arm]
        regrets[t - 1] = cum_regret
        realized[t - 1] = reward
        accepted[t - 1] = accept
        paid[t - 1] = p

    final_arm = np.array([int(np.argmax(counts[b])) for b in range(n_buckets)])
    return regrets, realized, accepted, paid, final_arm


def run_thompson_sampling(complexities, rng_alg, rng_outcome):
    mu0 = 0.0
    sigma0_sq = 1e6  # very wide / uninformative prior
    post_mean = np.full((n_buckets, n_arms), mu0)
    post_var = np.full((n_buckets, n_arms), sigma0_sq)
    obs_var = sigma_obs ** 2

    regrets = np.zeros(n_rounds)
    realized = np.zeros(n_rounds)
    accepted = np.zeros(n_rounds, dtype=bool)
    paid = np.zeros(n_rounds)
    cum_regret = 0.0
    pull_counts = np.zeros((n_buckets, n_arms))

    for t in range(1, n_rounds + 1):
        s = complexities[t - 1]
        b = bucket_of(s)
        samples = rng_alg.normal(post_mean[b], np.sqrt(post_var[b]))
        arm = int(np.argmax(samples))

        p = arm_grid[arm]
        pa = accept_prob(p, s)
        accept = rng_outcome.random() < pa
        reward = (value_per_order - base_price - p) if accept else -reject_penalty

        # conjugate Normal-Normal update (known observation variance)
        prior_mean = post_mean[b, arm]
        prior_var = post_var[b, arm]
        new_var = 1.0 / (1.0 / prior_var + 1.0 / obs_var)
        new_mean = new_var * (prior_mean / prior_var + reward / obs_var)
        post_mean[b, arm] = new_mean
        post_var[b, arm] = new_var
        pull_counts[b, arm] += 1

        cum_regret += oracle_er[b] - true_er[b, arm]
        regrets[t - 1] = cum_regret
        realized[t - 1] = reward
        accepted[t - 1] = accept
        paid[t - 1] = p

    final_arm = np.array([int(np.argmax(post_mean[b])) for b in range(n_buckets)])
    return regrets, realized, accepted, paid, final_arm


if st.button("Запустить симуляцию", type="primary"):
    rng_orders = np.random.default_rng(seed)
    complexities = simulate_orders(rng_orders, n_rounds)

    rng_alg_eg = np.random.default_rng(seed + 1)
    rng_out_eg = np.random.default_rng(seed + 2)
    rng_alg_ucb = np.random.default_rng(seed + 1)
    rng_out_ucb = np.random.default_rng(seed + 2)
    rng_alg_ts = np.random.default_rng(seed + 1)
    rng_out_ts = np.random.default_rng(seed + 2)

    regret_eg, realized_eg, accepted_eg, paid_eg, final_arm_eg = run_epsilon_greedy(
        complexities, rng_alg_eg, rng_out_eg
    )
    regret_ucb, realized_ucb, accepted_ucb, paid_ucb, final_arm_ucb = run_ucb1(
        complexities, rng_alg_ucb, rng_out_ucb
    )
    regret_ts, realized_ts, accepted_ts, paid_ts, final_arm_ts = run_thompson_sampling(
        complexities, rng_alg_ts, rng_out_ts
    )

    st.header("Накопленный регрет")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(regret_eg, label=f"epsilon-greedy (ε={epsilon})")
    ax.plot(regret_ucb, label="UCB1")
    ax.plot(regret_ts, label="Thompson Sampling (Gaussian)")
    ax.set_xlabel("Раунд t (заказ)")
    ax.set_ylabel("Накопленный регрет R(t)")
    ax.set_title("Накопленный регрет по стратегиям выбора надбавки")
    ax.legend(loc="best")
    st.pyplot(fig)

    st.header("Сходимость к оракульной надбавке по бакетам сложности")
    rows = []
    for b in range(n_buckets):
        rows.append(
            {
                "Бакет сложности": bucket_labels[b],
                "Оракульная надбавка": round(float(arm_grid[oracle_arm_idx[b]]), 1),
                "Epsilon-greedy сошёлся к": round(float(arm_grid[final_arm_eg[b]]), 1),
                "UCB1 сошёлся к": round(float(arm_grid[final_arm_ucb[b]]), 1),
                "Thompson Sampling сошёлся к": round(float(arm_grid[final_arm_ts[b]]), 1),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.header("Сводные метрики по стратегиям")

    def summary(realized, accepted, paid):
        return {
            "Суммарная реализованная прибыль": round(float(realized.sum()), 1),
            "Доля принятых заказов": f"{accepted.mean():.1%}",
            "Средняя выплаченная надбавка": round(float(paid.mean()), 1),
        }

    summary_df = pd.DataFrame(
        {
            f"Epsilon-greedy (ε={epsilon})": summary(realized_eg, accepted_eg, paid_eg),
            "UCB1": summary(realized_ucb, accepted_ucb, paid_ucb),
            "Thompson Sampling": summary(realized_ts, accepted_ts, paid_ts),
        }
    )
    st.dataframe(summary_df, use_container_width=True)

    st.markdown(
        f"""
**Интерпретация результатов.** Чем медленнее и более сублинейно растёт кривая накопленного
регрета, тем эффективнее стратегия находит для каждого бакета сложности надбавку, близкую к
оракульной. Как правило, UCB1 и Thompson Sampling показывают похожий или лучший результат по
сравнению с epsilon-greedy на длинных горизонтах (большое T), так как epsilon-greedy продолжает
с фиксированной вероятностью $\\varepsilon$ выбирать случайные (часто неоптимальные) надбавки
даже после того, как лучшая надбавка для бакета уже надёжно определена — это даёт линейный
«хвост» регрета.

При очень **низкой temperature** (резкий переход accept/reject) сигнал о том, какая надбавка
работает, становится почти бинарным и более «контрастным» — это упрощает различение хорошей и
плохой надбавки, но также делает награду более похожей на классическую Bernoulli-задачу с
резким порогом, и небольшая ошибка в оценке порога может стоить почти всех заказов в этой точке.
При **высокой temperature** (плавный переход) различия в ожидаемой награде между соседними
уровнями надбавки сглаживаются, регрет от выбора чуть неоптимальной надбавки меньше, но и
информации для быстрого обучения тоже меньше — алгоритмам требуется больше раундов, чтобы
надёжно различить близкие по качеству руки.

При **малом числе раундов T** относительно числа бакетов и рук многие (бакет, рука) комбинации
почти не успевают быть опробованы — особенно для бакетов с малой долей трафика (если сложность
распределена неравномерно) — и регрет остаётся высоким из-за недостатка исследования. При
**большом T** все три алгоритма обычно сходятся к надбавке, близкой к оракульной, для каждого
бакета, но UCB1 и Thompson Sampling делают это с меньшим накопленным регретом, так как их
исследование целенаправленно: они быстрее перестают пробовать заведомо плохие надбавки и
переключаются на эксплуатацию найденной лучшей.
"""
    )
else:
    st.info("Нажмите кнопку выше, чтобы запустить симуляцию и сравнить стратегии.")
