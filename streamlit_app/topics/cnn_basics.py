import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("9. Сверточные нейронные сети (CNN): свёртки, пулинг, рецептивное поле")
st.caption("Теория и формулы — pages/deep-learning.html, раздел 9.")

st.markdown(
    r"""
Свёртка скользящим окном вычисляет взвешенную сумму локальной области входа с ядром:
$$Y_{i,j} = \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} K_{m,n}\,X_{i+m,j+n} + b$$

Размер выходной карты при паддинге $p$ и шаге $s$:
$$H_{out} = \left\lfloor \frac{H+2p-k}{s}\right\rfloor + 1$$

Рецептивное поле растёт рекуррентно с каждым слоем:
$$r_l = r_{l-1} + (k_l-1)\prod_{i=1}^{l-1}s_i$$
"""
)

st.header("1. Применение свёрточного ядра к изображению")

img_choice = st.selectbox(
    "Тестовое изображение", ["Вертикальная грань", "Горизонтальная грань", "Диагональ + шум", "Шахматная доска"]
)

size_img = 28

def make_image(choice):
    img = np.zeros((size_img, size_img))
    if choice == "Вертикальная грань":
        img[:, size_img // 2:] = 1.0
    elif choice == "Горизонтальная грань":
        img[size_img // 2:, :] = 1.0
    elif choice == "Диагональ + шум":
        for i in range(size_img):
            for j in range(size_img):
                if i > j:
                    img[i, j] = 1.0
        rng = np.random.default_rng(0)
        img += rng.normal(0, 0.1, size=img.shape)
    else:  # chessboard
        for i in range(size_img):
            for j in range(size_img):
                img[i, j] = 1.0 if (i // 4 + j // 4) % 2 == 0 else 0.0
    return img

image = make_image(img_choice)

kernel_choice = st.selectbox(
    "Ядро (фильтр) 3×3",
    [
        "Sobel вертикальный (детектор вертикальных граней)",
        "Sobel горизонтальный (детектор горизонтальных граней)",
        "Размытие (box blur)",
        "Резкость (sharpen)",
        "Свой (задать вручную)",
    ],
)

kernels = {
    "Sobel вертикальный (детектор вертикальных граней)": np.array(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float
    ),
    "Sobel горизонтальный (детектор горизонтальных граней)": np.array(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float
    ),
    "Размытие (box blur)": np.ones((3, 3)) / 9,
    "Резкость (sharpen)": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float),
}

if kernel_choice == "Свой (задать вручную)":
    st.markdown("Введите 9 значений ядра 3×3:")
    cols_k = st.columns(3)
    vals = []
    default_vals = [0, 0, 0, 0, 1, 0, 0, 0, 0]
    for r in range(3):
        for c in range(3):
            idx = r * 3 + c
            with cols_k[c]:
                v = st.number_input(f"k[{r}][{c}]", value=float(default_vals[idx]), key=f"k_{r}_{c}")
            vals.append(v)
    kernel = np.array(vals).reshape(3, 3)
else:
    kernel = kernels[kernel_choice]

stride_cnn = st.slider("Stride", 1, 3, 1)
padding_cnn = st.selectbox("Padding", ["valid (без паддинга)", "same (паддинг=1)"])
pad = 1 if padding_cnn.startswith("same") else 0

def conv2d(img, k, stride=1, pad=0):
    if pad > 0:
        img = np.pad(img, pad, mode="constant")
    H, W = img.shape
    kh, kw = k.shape
    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            region = img[i * stride:i * stride + kh, j * stride:j * stride + kw]
            out[i, j] = np.sum(region * k)
    return out

feature_map = conv2d(image, kernel, stride_cnn, pad)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
axes[0].imshow(image, cmap="gray")
axes[0].set_title(f"Вход ({image.shape[0]}×{image.shape[1]})")
axes[0].axis("off")

axes[1].imshow(kernel, cmap="RdBu_r", vmin=-np.max(np.abs(kernel)), vmax=np.max(np.abs(kernel)))
for (r, c), v in np.ndenumerate(kernel):
    axes[1].text(c, r, f"{v:.1f}", ha="center", va="center", fontsize=9)
axes[1].set_title("Ядро (фильтр)")
axes[1].axis("off")

axes[2].imshow(feature_map, cmap="gray")
axes[2].set_title(f"Feature map ({feature_map.shape[0]}×{feature_map.shape[1]})")
axes[2].axis("off")
st.pyplot(fig)

st.caption(
    f"Формула выходного размера: H_out = floor((H + 2p - k)/s) + 1 = "
    f"floor(({size_img} + {2*pad} - 3)/{stride_cnn}) + 1 = {feature_map.shape[0]}."
)

st.header("2. Пулинг (Max / Average)")

pool_size = st.slider("Размер окна пулинга", 2, 4, 2)
pool_type = st.radio("Тип пулинга", ["Max pooling", "Average pooling"], horizontal=True)

def pool2d(img, size, mode="max"):
    H, W = img.shape
    out_h, out_w = H // size, W // size
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            region = img[i * size:(i + 1) * size, j * size:(j + 1) * size]
            out[i, j] = region.max() if mode == "max" else region.mean()
    return out

pooled = pool2d(np.abs(feature_map), pool_size, "max" if pool_type == "Max pooling" else "avg")

fig2, axes2 = plt.subplots(1, 2, figsize=(9, 4.5))
axes2[0].imshow(np.abs(feature_map), cmap="gray")
axes2[0].set_title("|Feature map| (вход пулинга)")
axes2[0].axis("off")
axes2[1].imshow(pooled, cmap="gray")
axes2[1].set_title(f"{pool_type}, окно {pool_size}×{pool_size} → {pooled.shape[0]}×{pooled.shape[1]}")
axes2[1].axis("off")
st.pyplot(fig2)

st.header("3. Рост рецептивного поля по глубине сети")

n_conv_layers = st.slider("Число свёрточных слоёв", 1, 12, 6)
kernel_size_rf = st.slider("Размер ядра в каждом слое", 1, 7, 3, 2)
stride_rf = st.slider("Stride в каждом слое (одинаковый для всех)", 1, 2, 1)

r = 1
strides_prod = 1
receptive_fields = [1]
for l in range(1, n_conv_layers + 1):
    r = r + (kernel_size_rf - 1) * strides_prod
    receptive_fields.append(r)
    strides_prod *= stride_rf

fig3, ax3 = plt.subplots(figsize=(9, 4))
ax3.plot(range(n_conv_layers + 1), receptive_fields, marker="o", color="purple")
ax3.set_xlabel("Номер слоя")
ax3.set_ylabel("Рецептивное поле (пикселей)")
ax3.set_title(f"Рост рецептивного поля: k={kernel_size_rf}, stride={stride_rf}")
ax3.grid(True, alpha=0.3)
st.pyplot(fig3)

st.success(
    f"После {n_conv_layers} слоёв нейрон в глубине сети «видит» область "
    f"{receptive_fields[-1]}×{receptive_fields[-1]} пикселей исходного изображения."
)
