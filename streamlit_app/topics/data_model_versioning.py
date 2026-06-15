import hashlib

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("3. Версионирование данных и моделей (DVC, Model Registry)")
st.caption("Теория и формулы — pages/mlops.html, раздел 3.")

st.markdown(
    r"""
DVC и аналоги используют принцип **content-addressable storage**: версия файла
идентифицируется хэшем его содержимого

$$ h = H(\text{content}) $$

Если содержимое не изменилось — хэш не меняется (дедупликация, повторная загрузка не нужна).
Если изменился хотя бы один байт — хэш меняется полностью, что позволяет однозначно проверить
целостность и версию файла.
"""
)

st.header("Демонстрация: как хэш реагирует на изменение данных")

st.markdown(
    "Введите содержимое \"датасета\" (например, CSV-строки) и посмотрите, как меняется его хэш "
    "при малейшем изменении."
)

col1, col2 = st.columns(2)
with col1:
    content_v1 = st.text_area(
        "Версия 1 датасета", "id,value\n1,10\n2,20\n3,30", height=120
    )
with col2:
    content_v2 = st.text_area(
        "Версия 2 датасета (попробуйте изменить одну цифру)", "id,value\n1,10\n2,20\n3,31", height=120
    )


def short_hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


h1 = short_hash(content_v1)
h2 = short_hash(content_v2)

st.code(f"hash(v1) = {h1}\nhash(v2) = {h2}")

if h1 == h2:
    st.success("Хэши совпадают — DVC посчитает это одной и той же версией данных, повторная загрузка не нужна.")
else:
    st.info(
        "Хэши полностью различны, хотя изменился, возможно, всего один символ — это нормальное "
        "свойство криптографических хэш-функций. DVC сохранит обе версии как разные объекты, и "
        "обе будут доступны для checkout по своему хэшу/коммиту."
    )

st.header("Симуляция: жизненный цикл версий в Model Registry")

st.markdown(
    """
Model Registry хранит версии модели с метаданными (метрика на валидации, стадия:
*None → Staging → Production → Archived*). Ниже — симуляция нескольких версий модели с разными
метриками; выберите, какая версия должна быть в Production.
"""
)

n_versions = st.slider("Число версий модели в registry", 2, 8, 5, 1)

rng = np.random.default_rng(123)
base_auc = 0.78
versions = []
for v in range(1, n_versions + 1):
    auc = base_auc + rng.normal(0, 0.01) + 0.005 * (v - 1) * (rng.random() < 0.7)
    versions.append({"version": v, "auc": round(float(auc), 4)})

best_idx = int(np.argmax([v["auc"] for v in versions]))

fig, ax = plt.subplots(figsize=(8, 4))
colors = ["seagreen" if i == best_idx else "steelblue" for i in range(n_versions)]
ax.bar([v["version"] for v in versions], [v["auc"] for v in versions], color=colors)
ax.set_xlabel("версия модели")
ax.set_ylabel("AUC на валидации")
ax.set_title("Версии модели в Model Registry (зелёная — лучшая по метрике)")
ax.set_ylim(min(v["auc"] for v in versions) - 0.01, max(v["auc"] for v in versions) + 0.01)
for v in versions:
    ax.text(v["version"], v["auc"] + 0.0008, f"{v['auc']:.4f}", ha="center", fontsize=8)
st.pyplot(fig)

st.markdown(f"**Версия с наилучшей метрикой: v{versions[best_idx]['version']}** (AUC = {versions[best_idx]['auc']:.4f})")

chosen_version = st.selectbox(
    "Выберите версию для перевода в стадию Production",
    options=[v["version"] for v in versions],
    index=best_idx,
)

if chosen_version == versions[best_idx]["version"]:
    st.success(
        f"v{chosen_version} → **Production**. Это версия с наилучшей валидационной метрикой — "
        "разумный выбор по умолчанию (но финальное решение должно подтверждаться A/B-тестом в проде)."
    )
else:
    chosen_auc = next(v["auc"] for v in versions if v["version"] == chosen_version)
    best_version = versions[best_idx]["version"]
    diff = versions[best_idx]["auc"] - chosen_auc
    st.warning(
        f"v{chosen_version} → **Production** (AUC = {chosen_auc:.4f}), хотя v{best_version} "
        f"имеет на {diff:.4f} выше offline-метрику. Это может быть осознанным решением "
        f"(например, v{chosen_version} проще/быстрее/стабильнее), но решение должно быть "
        "задокументировано в metadata registry."
    )

st.markdown(
    """
**Связка версий**: запись о версии модели в registry должна ссылаться на:
- Git commit, которым обучена модель,
- хэш/версию датасета (например, DVC-хэш), на котором она обучена,
- run эксперимент-трекера, из которого получены метрики.

Это замыкает цепочку: код + данные + run → версия модели в registry → стадия (Production/Staging/Archived).
"""
)
