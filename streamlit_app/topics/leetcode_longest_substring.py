import matplotlib.pyplot as plt
import streamlit as st

st.title("6.2 Longest Substring Without Repeating Characters")
st.caption("Условие и разбор решения — pages/leetcode.html, раздел 2.")

st.markdown(
    r"""
Дана строка `s`. Нужно найти длину самой длинной подстроки без повторяющихся символов.

**Пример:** `s = "abcabcbb"` → ответ `3` (подстрока `"abc"`).

**Идея решения:** скользящее окно `[left, right]` с хеш-таблицей "символ → последний
индекс". Двигаем `right` вправо; если символ уже встречался внутри текущего окна,
переставляем `left` сразу за предыдущим вхождением этого символа. На каждом шаге
обновляем ответ как `right - left + 1`.

- Время: $O(n)$
- Память: $O(\min(n, m))$, где $m$ — размер алфавита
"""
)

st.header("Демонстрация")

s = st.text_input("Строка s", "abcabcbb")

run = st.button("Найти самую длинную подстроку без повторов")

if run:
    if len(s) == 0:
        st.warning("Пустая строка — ответ 0.")
    else:
        last_seen = {}
        left = 0
        best = 0
        best_window = (0, 0)
        # запоминаем для каждого right состояние окна (left, right, best so far)
        history = []

        for right, ch in enumerate(s):
            if ch in last_seen and last_seen[ch] >= left:
                left = last_seen[ch] + 1
            last_seen[ch] = right
            cur_len = right - left + 1
            if cur_len > best:
                best = cur_len
                best_window = (left, right)
            history.append((left, right, cur_len))

        st.success(
            f"Длина самой длинной подстроки без повторов: **{best}** "
            f"(подстрока **\"{s[best_window[0]:best_window[1] + 1]}\"**, "
            f"индексы [{best_window[0]}, {best_window[1]}])"
        )

        # Визуализация: для каждого правого указателя показываем границы окна
        fig, ax = plt.subplots(figsize=(max(6, len(s) * 0.6), 3.5))

        for right, ch in enumerate(s):
            ax.text(right, 1.05, ch, ha="center", va="bottom", fontsize=12, fontweight="bold")
            ax.text(right, -0.15, str(right), ha="center", va="top", fontsize=8, color="gray")

        # Подсвечиваем итоговое лучшее окно
        l, r = best_window
        ax.add_patch(
            plt.Rectangle((l - 0.5, 0), r - l + 1, 1, facecolor="seagreen", alpha=0.3, edgecolor="seagreen", linewidth=2)
        )

        for i in range(len(s)):
            ax.add_patch(plt.Rectangle((i - 0.5, 0), 1, 1, fill=False, edgecolor="lightgray"))

        ax.set_xlim(-1, len(s))
        ax.set_ylim(-0.3, 1.3)
        ax.axis("off")
        ax.set_title(f"Лучшее окно без повторов: [{l}, {r}] -> \"{s[l:r + 1]}\" (длина {best})")
        st.pyplot(fig)

        # Таблица истории шагов
        st.subheader("Шаги алгоритма")
        st.dataframe(
            {
                "right": [h[1] for h in history],
                "символ": list(s),
                "left": [h[0] for h in history],
                "длина окна (right - left + 1)": [h[2] for h in history],
            },
            width="stretch",
        )

st.header("Код решения")
st.code(
    '''class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        last_seen = {}  # символ -> последний индекс, где он встретился
        left = 0
        best = 0

        for right, ch in enumerate(s):
            if ch in last_seen and last_seen[ch] >= left:
                left = last_seen[ch] + 1
            last_seen[ch] = right
            best = max(best, right - left + 1)

        return best
''',
    language="python",
)
