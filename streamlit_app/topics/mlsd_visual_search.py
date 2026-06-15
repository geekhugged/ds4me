import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.title("7.5 Визуальный поиск похожих товаров (e-commerce)")
st.caption(
    "Кейс — pages/ml-system-design-interview.html, раздел 5 "
    "(«Визуальный поиск похожих товаров (e-commerce)»)."
)

st.markdown(
    r"""
Визуальный поиск — задача **image retrieval**: embedding-модель (CNN/ViT, обученная через
metric/contrastive learning) отображает изображения товаров в общее векторное пространство так,
что похожие товары близки по косинусному расстоянию. На запросе — embedding фото пользователя,
поиск top-K ближайших соседей среди embedding'ов каталога (сотни миллионов) через приближённый
ANN-индекс (HNSW/FAISS), затем re-ranking с учётом категории, цены, бизнес-сигналов.

Ниже — упрощённая 2D-визуализация embedding-пространства: каждая точка — товар каталога с
координатами-эмбеддингом и категорией. Запрос пользователя — точка, для которой ищутся
ближайшие соседи (k-NN) по косинусному расстоянию.
"""
)

st.header("Симуляция: k-NN поиск в embedding-пространстве")

rng = np.random.default_rng(11)

categories = {
    "Сумки": (2, 2),
    "Обувь": (-2, 2),
    "Куртки": (2, -2),
    "Аксессуары": (-2, -2),
    "Платья": (0, 3),
}

n_per_category = st.slider("Товаров на категорию в каталоге (демо)", 10, 100, 40, 5)

rows = []
for cat, (cx, cy) in categories.items():
    pts = rng.normal(loc=(cx, cy), scale=0.9, size=(n_per_category, 2))
    for i, (x, y) in enumerate(pts):
        rows.append({"category": cat, "x": x, "y": y, "item_id": f"{cat}_{i}"})

catalog = pd.DataFrame(rows)

st.subheader("Запрос пользователя")
col1, col2, col3 = st.columns(3)
with col1:
    query_cat = st.selectbox("Похоже на категорию", list(categories.keys()))
with col2:
    noise = st.slider(
        "Domain gap (шум пользовательского фото)", 0.0, 3.0, 1.0, 0.1,
        help="Чем больше шум, тем дальше embedding пользовательского фото от 'чистых' каталожных фото",
    )
with col3:
    k = st.slider("k (число ближайших соседей)", 1, 20, 5)

cx, cy = categories[query_cat]
query_point = np.array([cx, cy]) + rng.normal(0, noise, size=2)

# косинусное расстояние (через нормализацию векторов)
def cosine_distance(a, b):
    a_norm = a / (np.linalg.norm(a) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return 1 - b_norm @ a_norm


catalog["cos_dist"] = cosine_distance(query_point, catalog[["x", "y"]].values)
catalog_sorted = catalog.sort_values("cos_dist").reset_index(drop=True)
top_k = catalog_sorted.head(k)

fig, ax = plt.subplots(figsize=(7, 7))
colors = plt.cm.tab10.colors
for i, (cat, _) in enumerate(categories.items()):
    subset = catalog[catalog["category"] == cat]
    ax.scatter(subset["x"], subset["y"], label=cat, alpha=0.4, color=colors[i % 10])

ax.scatter(top_k["x"], top_k["y"], facecolors="none", edgecolors="black", s=180, linewidths=2,
           label=f"top-{k} ближайших")
ax.scatter([query_point[0]], [query_point[1]], color="red", marker="*", s=300, label="запрос (фото пользователя)")
ax.set_title("Embedding-пространство товаров (2D-проекция)")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
ax.set_xlabel("dim 1")
ax.set_ylabel("dim 2")
st.pyplot(fig)

st.subheader(f"Top-{k} результатов визуального поиска")
st.dataframe(
    top_k[["item_id", "category", "cos_dist"]].style.format({"cos_dist": "{:.3f}"}),
    hide_index=True,
    use_container_width=True,
)

match_rate = (top_k["category"] == query_cat).mean()
st.metric(
    "Точность top-k (доля результатов из ожидаемой категории)",
    f"{match_rate:.1%}",
    help="Proxy для Precision@k — при росте domain gap (шума) точность падает",
)

st.markdown(
    """
**На что обратить внимание:**
- При увеличении `Domain gap` (шум) embedding пользовательского фото "уезжает" от кластера
  своей категории, и в top-k начинают попадать товары других категорий — это иллюстрирует
  проблему разрыва между "грязными" пользовательскими фото и студийными фото каталога.
- В реальной системе после ANN top-k проходит **re-ranking** более точной моделью с
  дополнительными сигналами (категория, бренд, цена, популярность), что отфильтровывает
  нерелевантные совпадения, видимые здесь.
"""
)

st.header("Back-of-the-envelope: индекс и serving")

col1, col2, col3 = st.columns(3)
with col1:
    catalog_size_m = st.slider("Размер каталога, млн товаров", 10, 1000, 500, 10)
with col2:
    emb_dim = st.slider("Размерность embedding", 64, 2048, 512, 64)
with col3:
    pq_bytes = st.slider("Размер вектора после Product Quantization, байт", 8, 256, 64, 8)

raw_size_gb = catalog_size_m * 1e6 * emb_dim * 4 / 1e9  # float32
pq_size_gb = catalog_size_m * 1e6 * pq_bytes / 1e9

m1, m2 = st.columns(2)
m1.metric("Размер индекса (float32, без сжатия)", f"{raw_size_gb:,.1f} ГБ")
m2.metric("Размер индекса (после PQ)", f"{pq_size_gb:,.1f} ГБ", delta=f"-{(1 - pq_size_gb/raw_size_gb):.0%}")

st.subheader("Latency budget")
col1, col2, col3, col4 = st.columns(4)
with col1:
    t_detect = st.slider("Object detection, мс", 0, 200, 70, 5)
with col2:
    t_embed = st.slider("Feature extraction, мс", 0, 200, 70, 5)
with col3:
    t_ann = st.slider("ANN-поиск, мс", 0, 100, 20, 5)
with col4:
    t_rerank = st.slider("Re-ranking, мс", 0, 200, 70, 5)

total_latency = t_detect + t_embed + t_ann + t_rerank
budget = st.slider("Latency budget p99, мс", 200, 1500, 800, 50)

st.metric(
    "Суммарная latency",
    f"{total_latency} мс",
    delta=f"{total_latency - budget:+d} мс относительно бюджета",
    delta_color="inverse",
)

if total_latency > budget:
    st.warning("Суммарная latency превышает бюджет — нужно оптимизировать самый дорогой этап или ослабить требование.")
else:
    st.success("Latency укладывается в бюджет.")
