import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("5. Контейнеризация ML-нагрузок: Docker и основы Kubernetes")
st.caption("Теория и формулы — pages/mlops.html, раздел 5.")

st.markdown(
    r"""
Kubernetes Horizontal Pod Autoscaler (HPA) рассчитывает желаемое число реплик по формуле:

$$ \text{desiredReplicas} = \left\lceil \text{currentReplicas} \times \frac{\text{currentUtilization}}{\text{targetUtilization}} \right\rceil $$

где $\text{currentUtilization}$ — наблюдаемая загрузка (например, средний % CPU по подам),
$\text{targetUtilization}$ — целевой уровень загрузки, заданный в конфигурации HPA.
"""
)

st.header("Калькулятор: Horizontal Pod Autoscaler")

col1, col2, col3 = st.columns(3)
with col1:
    current_replicas = st.slider("Текущее число реплик", 1, 20, 3, 1)
with col2:
    current_util = st.slider("Текущая загрузка CPU, %", 1, 300, 90, 1)
with col3:
    target_util = st.slider("Целевая загрузка CPU (target), %", 10, 100, 70, 5)

min_replicas = st.slider("Минимум реплик (minReplicas)", 1, 10, 1, 1)
max_replicas = st.slider("Максимум реплик (maxReplicas)", 1, 50, 10, 1)

desired_raw = current_replicas * (current_util / target_util)
desired = int(np.ceil(desired_raw))
desired_clamped = int(np.clip(desired, min_replicas, max_replicas))

st.markdown(
    f"""
$$ \\text{{desiredReplicas}} = \\lceil {current_replicas} \\times \\frac{{{current_util}}}{{{target_util}}} \\rceil = {desired} $$

После применения границ `minReplicas`/`maxReplicas`: **{desired_clamped} реплик(и)**.
"""
)

if desired_clamped == max_replicas and desired > max_replicas:
    st.warning(
        f"Расчётное число реплик ({desired}) превышает maxReplicas ({max_replicas}) — "
        "автоскейлер не сможет дальше масштабироваться, и сервис может начать деградировать "
        "по latency при дальнейшем росте нагрузки."
    )
elif desired_clamped == min_replicas and desired < min_replicas:
    st.info(
        f"Расчётное число реплик ({desired}) меньше minReplicas ({min_replicas}) — "
        "система держит минимум реплик независимо от нагрузки (например, для устойчивости к всплескам)."
    )

st.header("Симуляция: динамика автоскейлинга во времени")

st.markdown(
    "Смоделируем, как меняется нагрузка (запросов/сек) и как HPA подстраивает число реплик. "
    "Каждая реплика обрабатывает фиксированную пропускную способность при целевой загрузке."
)

col4, col5, col6 = st.columns(3)
with col4:
    capacity_per_replica = st.slider("Пропускная способность одной реплики при 100% CPU, RPS", 10, 200, 50, 5)
with col5:
    sim_minutes = st.slider("Длительность симуляции, минут", 10, 120, 60, 10)
with col6:
    traffic_pattern = st.radio("Паттерн трафика", ["Плавный рост", "Резкий всплеск (spike)", "Суточная волна"])

t = np.arange(sim_minutes)

if traffic_pattern == "Плавный рост":
    rps = 20 + 2.0 * t
elif traffic_pattern == "Резкий всплеск (spike)":
    rps = np.full(sim_minutes, 30.0)
    spike_start = sim_minutes // 3
    spike_end = spike_start + max(2, sim_minutes // 10)
    rps[spike_start:spike_end] = 250.0
else:
    rps = 100 + 80 * np.sin(2 * np.pi * t / sim_minutes)
    rps = np.clip(rps, 5, None)

# Симуляция HPA с задержкой реакции (HPA опрашивает метрики не мгновенно)
replicas = np.zeros(sim_minutes)
replicas[0] = current_replicas
reaction_delay = 1  # шаг задержки реакции HPA (в "тиках")

for i in range(1, sim_minutes):
    util = (rps[i - reaction_delay] / (replicas[i - 1] * capacity_per_replica)) * 100
    desired_i = np.ceil(replicas[i - 1] * (util / target_util))
    replicas[i] = np.clip(desired_i, min_replicas, max_replicas)

# фактическая загрузка с учётом числа реплик
actual_util = (rps / (replicas * capacity_per_replica)) * 100
saturated = actual_util > 100

fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

axes[0].plot(t, rps, color="steelblue", label="нагрузка, RPS")
axes[0].set_ylabel("RPS")
axes[0].legend(loc="upper left")
axes[0].set_title("Входящая нагрузка")

axes[1].step(t, replicas, where="post", color="seagreen", label="число реплик (HPA)")
ax_twin = axes[1].twinx()
ax_twin.plot(t, actual_util, color="darkorange", alpha=0.7, label="фактическая загрузка CPU, %")
ax_twin.axhline(target_util, color="gray", linestyle="--", label=f"target = {target_util}%")
ax_twin.axhline(100, color="crimson", linestyle=":", label="100% (насыщение)")
axes[1].set_ylabel("реплики")
ax_twin.set_ylabel("загрузка CPU, %")
axes[1].set_xlabel("время, мин")
axes[1].set_title("Реакция HPA и фактическая загрузка")

lines1, labels1 = axes[1].get_legend_handles_labels()
lines2, labels2 = ax_twin.get_legend_handles_labels()
axes[1].legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

st.pyplot(fig)

n_saturated = int(saturated.sum())
if n_saturated > 0:
    st.warning(
        f"В {n_saturated} из {sim_minutes} минут фактическая загрузка превышала 100% — "
        "сервис мог не успевать обрабатывать запросы (рост latency, возможные таймауты), "
        "пока HPA не добавил реплики. Это типичная проблема при резких всплесках и "
        "недостаточном `minReplicas` / медленном времени старта подов (cold start)."
    )
else:
    st.success(
        "Во все моменты времени фактическая загрузка оставалась в безопасных пределах — "
        "автоскейлер успевал подстраиваться под нагрузку."
    )

st.header("Размер Docker-образа: влияние multi-stage сборки")

st.markdown(
    "Для ML-сервисов размер образа напрямую влияет на скорость деплоя и автоскейлинга "
    "(новый под должен скачать образ перед запуском)."
)

base_size = st.slider("Размер базового образа (например, CUDA runtime), МБ", 200, 4000, 1200, 100)
deps_size = st.slider("Размер ML-зависимостей (torch, sklearn и т.д.), МБ", 200, 6000, 2500, 100)
build_tools_size = st.slider("Размер инструментов сборки (компиляторы, dev-заголовки), МБ", 0, 3000, 1500, 100)
code_size = st.slider("Размер кода приложения, МБ", 1, 200, 20, 1)

single_stage = base_size + deps_size + build_tools_size + code_size
multi_stage = base_size + deps_size + code_size  # build tools отбрасываются на финальном этапе

fig2, ax2 = plt.subplots(figsize=(7, 3.5))
ax2.barh(["Single-stage сборка", "Multi-stage сборка"], [single_stage, multi_stage], color=["crimson", "seagreen"])
ax2.set_xlabel("размер образа, МБ")
ax2.set_title("Размер финального образа")
for i, v in enumerate([single_stage, multi_stage]):
    ax2.text(v + 30, i, f"{v} МБ", va="center")
st.pyplot(fig2)

reduction = (1 - multi_stage / single_stage) * 100
st.markdown(
    f"Multi-stage сборка позволяет уменьшить размер финального образа на **{reduction:.0f}%** "
    f"({single_stage} МБ → {multi_stage} МБ), исключив из runtime-образа инструменты, нужные "
    "только на этапе сборки (компиляторы, dev-заголовки)."
)
