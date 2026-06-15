import matplotlib.pyplot as plt
import streamlit as st

st.title("6.5 Merge Intervals")
st.caption("Условие и разбор решения — pages/leetcode.html, раздел 5.")

st.markdown(
    r"""
Дан массив `intervals`, где `intervals[i] = [start_i, end_i]`. Нужно объединить все
пересекающиеся интервалы и вернуть массив непересекающихся интервалов, который
покрывает все интервалы из входа.

**Идея решения:** сортируем интервалы по началу — после сортировки все пересекающиеся
интервалы оказываются рядом друг с другом. Затем за один проход жадно объединяем
соседние интервалы: если `start` текущего интервала $\le$ `end` последнего объединённого,
расширяем последний объединённый интервал (`end = max(end, текущий end)`), иначе
добавляем новый.

- Время: $O(n \log n)$ — доминирует сортировка
- Память: $O(n)$ — на хранение результата
"""
)

st.header("Демонстрация")

default_intervals = "[1,3],[2,6],[8,10],[15,18]"
intervals_input = st.text_input("Интервалы (формат [a,b],[c,d],...)", default_intervals)

run = st.button("Объединить интервалы")


def parse_intervals(text: str):
    text = text.strip()
    if not text:
        return []
    result = []
    for part in text.replace(" ", "").split("],"):
        part = part.strip("[]")
        if not part:
            continue
        a, b = part.split(",")
        result.append([int(a), int(b)])
    return result


if run:
    try:
        intervals = parse_intervals(intervals_input)
        if not intervals or any(len(iv) != 2 or iv[0] > iv[1] for iv in intervals):
            raise ValueError
    except ValueError:
        st.error("Не удалось разобрать интервалы. Используйте формат [a,b],[c,d],... где a <= b.")
        intervals = None

    if intervals is not None:
        original = [iv[:] for iv in intervals]

        sorted_intervals = sorted(intervals, key=lambda iv: iv[0])

        merged = [sorted_intervals[0][:]]
        for start, end in sorted_intervals[1:]:
            last = merged[-1]
            if start <= last[1]:
                last[1] = max(last[1], end)
            else:
                merged.append([start, end])

        st.success(f"Результат: **{merged}**")

        # Визуализация: исходные интервалы сверху, объединённые снизу
        fig, ax = plt.subplots(figsize=(8, max(3, (len(original) + len(merged)) * 0.4)))

        for i, (start, end) in enumerate(sorted_intervals):
            y = len(original) - i
            ax.plot([start, end], [y, y], color="steelblue", linewidth=6, solid_capstyle="butt")
            ax.text((start + end) / 2, y + 0.15, f"[{start},{end}]", ha="center", fontsize=8)

        offset = len(original) + 1.5
        for i, (start, end) in enumerate(merged):
            y = offset + (len(merged) - i)
            ax.plot([start, end], [y, y], color="seagreen", linewidth=8, solid_capstyle="butt")
            ax.text((start + end) / 2, y + 0.15, f"[{start},{end}]", ha="center", fontsize=9, fontweight="bold")

        ax.text(-1, len(original) + 0.7, "Исходные (отсортированные):", fontsize=9, ha="left", color="steelblue")
        ax.text(-1, offset + len(merged) + 0.7, "Объединённые:", fontsize=9, ha="left", color="seagreen")

        all_vals = [v for iv in original for v in iv]
        ax.set_xlim(min(all_vals) - 1, max(all_vals) + 1)
        ax.set_ylim(0, offset + len(merged) + 1.5)
        ax.axis("off")
        ax.set_title("Объединение пересекающихся интервалов")
        st.pyplot(fig)

st.header("Код решения")
st.code(
    '''from typing import List

class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if not intervals:
            return []

        intervals.sort(key=lambda interval: interval[0])

        merged = [intervals[0][:]]
        for start, end in intervals[1:]:
            last = merged[-1]
            if start <= last[1]:
                last[1] = max(last[1], end)
            else:
                merged.append([start, end])

        return merged
''',
    language="python",
)
