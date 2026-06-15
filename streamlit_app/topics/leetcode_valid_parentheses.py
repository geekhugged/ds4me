import matplotlib.pyplot as plt
import streamlit as st

st.title("6.3 Valid Parentheses")
st.caption("Условие и разбор решения — pages/leetcode.html, раздел 3.")

st.markdown(
    r"""
Дана строка `s`, состоящая только из символов `(`, `)`, `{`, `}`, `[`, `]`. Нужно
определить, является ли строка корректной последовательностью скобок: каждая
открывающая скобка должна закрываться скобкой того же типа в правильном порядке
(последняя открытая — первая закрытая).

**Идея решения:** стек (LIFO). Открывающие скобки кладём в стек; встречая закрывающую —
проверяем, что верхушка стека — соответствующая открывающая, и снимаем её. Строка
корректна тогда и только тогда, когда в конце стек пуст.

- Время: $O(n)$
- Память: $O(n)$
"""
)

st.header("Демонстрация")

s = st.text_input("Строка s", "{[()]}")

run = st.button("Проверить корректность")

if run:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    history = []  # (символ, действие, стек после операции, ошибка?)
    error_at = None

    for i, ch in enumerate(s):
        if ch not in "()[]{}":
            error_at = i
            history.append((ch, "недопустимый символ", list(stack), True))
            break
        if ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                history.append((ch, "ошибка: нет подходящей открывающей", list(stack), True))
                error_at = i
                break
            stack.pop()
            history.append((ch, "снять со стека", list(stack), False))
        else:
            stack.append(ch)
            history.append((ch, "положить в стек", list(stack), False))

    is_valid = error_at is None and not stack

    if is_valid:
        st.success("Строка корректна: True")
    else:
        if error_at is not None:
            st.error(f"Строка некорректна: False (несоответствие на позиции {error_at}, символ '{s[error_at]}')")
        else:
            st.error(f"Строка некорректна: False (после обработки в стеке остались незакрытые скобки: {stack})")

    # Таблица шагов
    st.subheader("Шаги алгоритма")
    st.dataframe(
        {
            "позиция": list(range(len(history))),
            "символ": [h[0] for h in history],
            "действие": [h[1] for h in history],
            "стек после операции": ["".join(h[2]) if h[2] else "(пусто)" for h in history],
        },
        width="stretch",
    )

    # Визуализация стека на каждом шаге как столбчатая диаграмма глубины
    fig, ax = plt.subplots(figsize=(max(6, len(history) * 0.6), 3.5))
    depths = [len(h[2]) for h in history]
    colors = ["crimson" if h[3] else "seagreen" for h in history]
    ax.bar(range(len(history)), depths, color=colors, edgecolor="black")
    for i, h in enumerate(history):
        ax.text(i, depths[i] + 0.05, h[0], ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(history)))
    ax.set_xlabel("шаг")
    ax.set_ylabel("глубина стека")
    ax.set_title("Глубина стека после каждой операции (красный — ошибка)")
    st.pyplot(fig)

st.header("Код решения")
st.code(
    '''class Solution:
    def isValid(self, s: str) -> bool:
        pairs = {')': '(', ']': '[', '}': '{'}
        stack = []

        for ch in s:
            if ch in pairs:  # закрывающая скобка
                if not stack or stack[-1] != pairs[ch]:
                    return False
                stack.pop()
            else:  # открывающая скобка
                stack.append(ch)

        return not stack
''',
    language="python",
)
