from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.title("6.4 Number of Islands")
st.caption("Условие и разбор решения — pages/leetcode.html, раздел 4.")

st.markdown(
    r"""
Дана 2D-сетка `grid` размером `m x n` из символов `'1'` (земля) и `'0'` (вода). Нужно
вернуть количество "островов" — групп из единиц, соединённых горизонтально или
вертикально.

**Идея решения:** подсчёт компонент связности. Идём по всем клеткам; как только
встречаем непосещённую клетку `'1'` — это новый остров, запускаем BFS/DFS, который
помечает все связанные с ней клетки земли как посещённые.

- Время: $O(m \times n)$
- Память: $O(m \times n)$ в худшем случае (размер очереди BFS)
"""
)

st.header("Демонстрация")

st.markdown(
    "Введите сетку: каждая строка — последовательность из `0` и `1` (без пробелов или с пробелами/запятыми)."
)

default_grid = "11000\n11000\n00100\n00011"
grid_text = st.text_area("Сетка grid", default_grid, height=120)

run = st.button("Найти острова")

if run:
    rows_raw = [line.strip() for line in grid_text.splitlines() if line.strip() != ""]
    # допускаем разделители: пробелы, запятые или просто слитные цифры
    parsed_rows = []
    valid = True
    for line in rows_raw:
        cleaned = line.replace(",", " ").split()
        if len(cleaned) == 1 and len(cleaned[0]) > 1:
            cells = list(cleaned[0])
        else:
            cells = cleaned
        if not all(c in ("0", "1") for c in cells):
            valid = False
            break
        parsed_rows.append(cells)

    if not valid or not parsed_rows:
        st.error("Не удалось разобрать сетку — используйте только символы 0 и 1.")
    elif len({len(r) for r in parsed_rows}) != 1:
        st.error("Все строки сетки должны быть одинаковой длины.")
    else:
        grid = [row[:] for row in parsed_rows]
        rows, cols = len(grid), len(grid[0])
        island_id = [[0] * cols for _ in range(rows)]
        islands = 0

        work = [row[:] for row in grid]

        for r in range(rows):
            for c in range(cols):
                if work[r][c] == "1":
                    islands += 1
                    queue = deque([(r, c)])
                    work[r][c] = "0"
                    island_id[r][c] = islands
                    while queue:
                        cr, cc = queue.popleft()
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < rows and 0 <= nc < cols and work[nr][nc] == "1":
                                work[nr][nc] = "0"
                                island_id[nr][nc] = islands
                                queue.append((nr, nc))

        st.success(f"Количество островов: **{islands}**")

        # Визуализация сетки с раскраской по острову
        fig, ax = plt.subplots(figsize=(cols * 0.7, rows * 0.7))
        display_grid = np.array(
            [[island_id[r][c] if grid[r][c] == "1" else 0 for c in range(cols)] for r in range(rows)]
        )
        cmap = plt.get_cmap("tab20")
        ax.imshow(display_grid, cmap=cmap, vmin=0, vmax=max(20, islands))

        for r in range(rows):
            for c in range(cols):
                label = grid[r][c]
                ax.text(c, r, label, ha="center", va="center", fontsize=12, color="black")
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="white", linewidth=1))

        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.set_title(f"Сетка ({rows}x{cols}): найдено {islands} остров(ов), цвет = id острова")
        st.pyplot(fig)

st.header("Код решения")
st.code(
    '''from collections import deque
from typing import List

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid or not grid[0]:
            return 0

        rows, cols = len(grid), len(grid[0])
        islands = 0

        def bfs(start_r: int, start_c: int) -> None:
            queue = deque([(start_r, start_c)])
            grid[start_r][start_c] = '0'  # помечаем как посещённую
            while queue:
                r, c = queue.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == '1':
                        grid[nr][nc] = '0'
                        queue.append((nr, nc))

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == '1':
                    islands += 1
                    bfs(r, c)

        return islands
''',
    language="python",
)
