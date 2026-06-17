import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("16. Многорукие бандиты: explore-exploit, regret, epsilon-greedy, UCB1, Thompson Sampling")
st.caption("Теория и формулы — pages/ml-theory.html, раздел 16.")

st.markdown(
    r"""
Регрет за $T$ шагов: $R(T) = \sum_{t=1}^{T} (\mu^* - \mu_{a_t})$, где $\mu^*$ — среднее
вознаграждение оптимальной руки, а $\mu_{a_t}$ — среднее вознаграждение выбранной на шаге $t$ руки.

**Epsilon-greedy**: с вероятностью $1-\varepsilon$ выбираем руку с максимальной оценкой
$\hat{\mu}_i(t)$, с вероятностью $\varepsilon$ — случайную руку.

**UCB1**: $a_t = \arg\max_i \left(\hat{\mu}_i(t) + \sqrt{\dfrac{2\ln t}{n_i(t)}}\right)$ —
выбираем руку с максимальной верхней доверительной границей.

**Thompson Sampling (Beta-Bernoulli)**: для каждой руки $i$ сэмплируем
$\tilde{\theta}_i \sim \text{Beta}(\alpha_i, \beta_i)$ и выбираем $a_t = \arg\max_i \tilde{\theta}_i$;
после наблюдения награды $r_t$ обновляем $\alpha_{a_t} \leftarrow \alpha_{a_t} + r_t$,
$\beta_{a_t} \leftarrow \beta_{a_t} + (1 - r_t)$.
"""
)

st.header("Интерактивная симуляция бандита")

col1, col2 = st.columns(2)
with col1:
    n_arms = st.slider("Число рук K", 2, 10, 5, 1)
    n_rounds = st.slider("Число раундов T", 100, 5000, 1000, 100)
with col2:
    epsilon = st.slider("epsilon (для epsilon-greedy)", 0.0, 1.0, 0.1, 0.01)
    seed = st.slider("Random seed", 0, 100, 42, 1)

rng = np.random.default_rng(seed)
true_probs = rng.uniform(0.05, 0.95, size=n_arms)
true_probs = np.round(true_probs, 2)
best_arm = int(np.argmax(true_probs))
mu_star = true_probs[best_arm]

st.write("Истинные вероятности успеха (Bernoulli) для каждой руки (агенту неизвестны):")
st.write({f"рука {i}": p for i, p in enumerate(true_probs)})
st.write(f"Оптимальная рука: **{best_arm}** (μ* = {mu_star:.2f})")


def run_epsilon_greedy(true_probs, n_rounds, epsilon, rng):
    K = len(true_probs)
    counts = np.zeros(K)
    sums = np.zeros(K)
    regrets = np.zeros(n_rounds)
    cum_regret = 0.0
    mu_star = true_probs.max()
    for t in range(1, n_rounds + 1):
        if rng.random() < epsilon or np.any(counts == 0):
            if np.any(counts == 0):
                arm = int(np.argmin(counts))
            else:
                arm = rng.integers(0, K)
        else:
            means = sums / np.maximum(counts, 1)
            arm = int(np.argmax(means))
        reward = 1.0 if rng.random() < true_probs[arm] else 0.0
        counts[arm] += 1
        sums[arm] += reward
        cum_regret += mu_star - true_probs[arm]
        regrets[t - 1] = cum_regret
    return regrets


def run_ucb1(true_probs, n_rounds, rng):
    K = len(true_probs)
    counts = np.zeros(K)
    sums = np.zeros(K)
    regrets = np.zeros(n_rounds)
    cum_regret = 0.0
    mu_star = true_probs.max()
    for t in range(1, n_rounds + 1):
        if np.any(counts == 0):
            arm = int(np.argmin(counts))
        else:
            means = sums / counts
            bonus = np.sqrt(2 * np.log(t) / counts)
            arm = int(np.argmax(means + bonus))
        reward = 1.0 if rng.random() < true_probs[arm] else 0.0
        counts[arm] += 1
        sums[arm] += reward
        cum_regret += mu_star - true_probs[arm]
        regrets[t - 1] = cum_regret
    return regrets


def run_thompson_sampling(true_probs, n_rounds, rng):
    K = len(true_probs)
    alpha = np.ones(K)
    beta = np.ones(K)
    regrets = np.zeros(n_rounds)
    cum_regret = 0.0
    mu_star = true_probs.max()
    for t in range(1, n_rounds + 1):
        samples = rng.beta(alpha, beta)
        arm = int(np.argmax(samples))
        reward = 1.0 if rng.random() < true_probs[arm] else 0.0
        if reward > 0:
            alpha[arm] += 1
        else:
            beta[arm] += 1
        cum_regret += mu_star - true_probs[arm]
        regrets[t - 1] = cum_regret
    return regrets


if st.button("Запустить сравнение стратегий", type="primary"):
    rng_eg = np.random.default_rng(seed)
    rng_ucb = np.random.default_rng(seed)
    rng_ts = np.random.default_rng(seed)

    regret_eg = run_epsilon_greedy(true_probs, n_rounds, epsilon, rng_eg)
    regret_ucb = run_ucb1(true_probs, n_rounds, rng_ucb)
    regret_ts = run_thompson_sampling(true_probs, n_rounds, rng_ts)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(regret_eg, label=f"epsilon-greedy (ε={epsilon})")
    ax.plot(regret_ucb, label="UCB1")
    ax.plot(regret_ts, label="Thompson Sampling")
    ax.set_xlabel("Раунд t")
    ax.set_ylabel("Накопленный регрет R(t)")
    ax.set_title("Сравнение накопленного регрета стратегий бандита")
    ax.legend(loc="best")
    st.pyplot(fig)

    st.write("Итоговый накопленный регрет после всех раундов:")
    st.write(
        {
            f"epsilon-greedy (ε={epsilon})": float(regret_eg[-1]),
            "UCB1": float(regret_ucb[-1]),
            "Thompson Sampling": float(regret_ts[-1]),
        }
    )

    st.markdown(
        """
Чем медленнее растёт кривая накопленного регрета (особенно на больших t), тем эффективнее
стратегия — в идеале регрет должен расти сублинейно (замедляться со временем), что означает,
что алгоритм всё чаще выбирает оптимальную руку. Постоянный epsilon (без затухания) обычно
даёт линейный рост регрета на больших горизонтах, так как агент продолжает с фиксированной
вероятностью выбирать случайные (часто неоптимальные) руки даже после того, как лучшая рука уже
надёжно определена. UCB1 и Thompson Sampling, как правило, показывают сублинейный
(логарифмический) рост регрета и обгоняют epsilon-greedy на длинных горизонтах — попробуйте
увеличить число раундов T, чтобы это увидеть нагляднее.
"""
    )
else:
    st.info("Нажмите кнопку выше, чтобы запустить симуляцию и сравнить стратегии.")
