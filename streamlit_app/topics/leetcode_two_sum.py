import matplotlib.pyplot as plt
import streamlit as st

st.title("6.1 Two Sum")
st.caption("Условие и разбор решения — pages/leetcode.html, раздел 1.")

st.markdown(
    r"""
Дан массив целых чисел `nums` и целое число `target`. Нужно вернуть индексы двух чисел из
массива, сумма которых равна `target` (предполагается, что решение существует и единственно,
а один элемент нельзя использовать дважды).

**Идея решения:** один проход по массиву с хеш-таблицей "значение → индекс". Для каждого
элемента `x` проверяем, есть ли в таблице `target - x`. Если есть — нашли пару. Если нет —
добавляем `x` в таблицу и идём дальше.

- Время: $O(n)$
- Память: $O(n)$
"""
)

st.header("Демонстрация")

col1, col2 = st.columns(2)
with col1:
    nums_input = st.text_input("Массив nums (числа через запятую)", "2, 7, 11, 15")
with col2:
    target = st.number_input("target", value=9, step=1)

run = st.button("Найти пару индексов")

if run:
    try:
        nums = [int(x.strip()) for x in nums_input.split(",") if x.strip() != ""]
    except ValueError:
        st.error("Не удалось разобрать массив — убедитесь, что это числа через запятую.")
        nums = None

    if nums is not None:
        if len(nums) < 2:
            st.error("В массиве должно быть хотя бы 2 элемента.")
        else:
            seen = {}
            result = None
            steps = []  # для визуализации: (индекс, значение, статус)

            for i, x in enumerate(nums):
                complement = target - x
                if complement in seen:
                    result = [seen[complement], i]
                    steps.append((i, x, "found"))
                    break
                seen[x] = i
                steps.append((i, x, "stored"))

            if result is not None:
                st.success(
                    f"Найдена пара индексов: **{result}** "
                    f"(nums[{result[0]}] + nums[{result[1]}] = "
                    f"{nums[result[0]]} + {nums[result[1]]} = {target})"
                )
            else:
                st.warning("Пара с такой суммой не найдена в данном массиве.")

            # Визуализация прохода по массиву
            fig, ax = plt.subplots(figsize=(max(6, len(nums) * 0.8), 3))
            colors = []
            for i, x in enumerate(nums):
                if result and i in result:
                    colors.append("seagreen")
                elif any(s[0] == i for s in steps):
                    colors.append("lightsteelblue")
                else:
                    colors.append("lightgray")

            ax.bar(range(len(nums)), nums, color=colors, edgecolor="black")
            for i, x in enumerate(nums):
                ax.text(i, x, str(x), ha="center", va="bottom", fontsize=10)
                ax.text(i, 0, f"i={i}", ha="center", va="top", fontsize=8, color="gray")

            ax.set_title("Массив nums (зелёным — найденная пара)")
            ax.set_xticks(range(len(nums)))
            ax.axhline(0, color="black", linewidth=0.5)
            st.pyplot(fig)

st.header("Код решения")
st.code(
    '''from typing import List

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        seen = {}  # значение -> индекс
        for i, x in enumerate(nums):
            complement = target - x
            if complement in seen:
                return [seen[complement], i]
            seen[x] = i
        return []
''',
    language="python",
)
