import random

import streamlit as st

st.title("6. Git и GitHub: основы и рабочий процесс")
st.caption("Теория — pages/mlops.html, раздел 6.")

st.markdown(
    r"""
Git хранит историю проекта как граф коммитов: каждый коммит — узел со ссылкой на
своего "родителя", а ветка — просто именованный указатель на один из таких узлов.
Понимание этого графа — ключ к пониманию команд `branch`, `merge`, `rebase` и
`checkout`/`switch`.

Эта страница — две интерактивные песочницы:

1. **Граф коммитов и веток** — пошагово создавайте коммиты, ветки и слияния и
   смотрите, как меняется граф.
2. **Тренажёр команд** — типичные сценарии работы, нужно выбрать правильную
   команду Git.
"""
)

# ---------------------------------------------------------------------------
# 1. Граф коммитов и веток
# ---------------------------------------------------------------------------

st.header("1. Граф коммитов и веток")

st.markdown(
    "Состояние репозитория хранится в виде списка коммитов (с родителями) и "
    "указателей-веток на конкретные коммиты. `HEAD` указывает на текущую ветку."
)

if "commits" not in st.session_state:
    # commit_id -> parent_id (None для первого коммита)
    st.session_state.commits = {"C1": None}
    st.session_state.branches = {"main": "C1"}
    st.session_state.current_branch = "main"
    st.session_state.next_commit_num = 2
    st.session_state.log = ["Инициализирован репозиторий: создан коммит C1 в ветке main."]


def add_commit():
    cur_branch = st.session_state.current_branch
    parent = st.session_state.branches[cur_branch]
    new_id = f"C{st.session_state.next_commit_num}"
    st.session_state.commits[new_id] = parent
    st.session_state.branches[cur_branch] = new_id
    st.session_state.next_commit_num += 1
    st.session_state.log.append(
        f"`git commit` в ветке `{cur_branch}`: создан коммит {new_id} "
        f"(родитель: {parent})."
    )


def create_branch(name):
    if not name:
        st.session_state.log.append("Имя ветки не может быть пустым.")
        return
    if name in st.session_state.branches:
        st.session_state.log.append(f"Ветка `{name}` уже существует.")
        return
    cur_branch = st.session_state.current_branch
    head = st.session_state.branches[cur_branch]
    st.session_state.branches[name] = head
    st.session_state.current_branch = name
    st.session_state.log.append(
        f"`git switch -c {name}`: создана новая ветка `{name}` на коммите {head} "
        f"и выполнено переключение на неё (HEAD -> {name})."
    )


def switch_branch(name):
    st.session_state.current_branch = name
    st.session_state.log.append(f"`git switch {name}`: HEAD теперь указывает на ветку `{name}`.")


def merge_branch(source, target):
    if source == target:
        st.session_state.log.append("Нельзя слить ветку саму с собой.")
        return
    source_head = st.session_state.branches[source]
    target_head = st.session_state.branches[target]

    # является ли target предком source (fast-forward)?
    def ancestors(commit_id):
        path = []
        cur = commit_id
        while cur is not None:
            path.append(cur)
            cur = st.session_state.commits[cur]
        return path

    if target_head in ancestors(source_head) and target_head != source_head:
        # fast-forward merge
        st.session_state.branches[target] = source_head
        st.session_state.log.append(
            f"`git switch {target}` затем `git merge {source}`: fast-forward merge — "
            f"ветка `{target}` просто передвинута на {source_head}, новый коммит не создан."
        )
    elif source_head == target_head:
        st.session_state.log.append(
            f"Ветки `{source}` и `{target}` уже указывают на один коммит — нечего слиять."
        )
    else:
        # merge commit с двумя родителями (упрощённо: храним только первого родителя в commits,
        # но в логе и графе показываем оба родителя)
        new_id = f"M{st.session_state.next_commit_num}"
        st.session_state.commits[new_id] = target_head
        st.session_state.merge_parents = getattr(st.session_state, "merge_parents", {})
        st.session_state.merge_parents[new_id] = source_head
        st.session_state.branches[target] = new_id
        st.session_state.next_commit_num += 1
        st.session_state.log.append(
            f"`git switch {target}` затем `git merge {source}`: создан merge-коммит {new_id} "
            f"с двумя родителями ({target_head} и {source_head}). Если изменения "
            "пересекались, перед этим нужно было разрешить конфликт."
        )


col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown(f"**Текущая ветка (HEAD):** `{st.session_state.current_branch}`")
    if st.button("git commit  (новый коммит в текущей ветке)"):
        add_commit()

with col_b:
    new_branch_name = st.text_input("Имя новой ветки", value="feature/new-model")
    if st.button("git switch -c <ветка>  (создать и переключиться)"):
        create_branch(new_branch_name)

with col_c:
    branch_to_switch = st.selectbox(
        "Переключиться на ветку", sorted(st.session_state.branches.keys())
    )
    if st.button("git switch <ветка>"):
        switch_branch(branch_to_switch)

st.markdown("**Слияние веток**")
col_d, col_e, col_f = st.columns(3)
branch_names = sorted(st.session_state.branches.keys())
with col_d:
    merge_source = st.selectbox("Влить из ветки (source)", branch_names, key="merge_source")
with col_e:
    merge_target = st.selectbox(
        "В ветку (target)",
        branch_names,
        index=branch_names.index("main") if "main" in branch_names else 0,
        key="merge_target",
    )
with col_f:
    st.markdown("&nbsp;")
    if st.button("git merge <source> в <target>"):
        merge_branch(merge_source, merge_target)

if st.button("Сбросить граф к начальному состоянию"):
    for key in ["commits", "branches", "current_branch", "next_commit_num", "log", "merge_parents"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# Построение DOT-графа
dot_lines = ["digraph G {", "  rankdir=LR;", "  node [shape=circle, style=filled, fillcolor=lightblue, fontsize=11];"]

merge_parents = getattr(st.session_state, "merge_parents", {})
for commit_id, parent_id in st.session_state.commits.items():
    if parent_id is not None:
        dot_lines.append(f'  "{parent_id}" -> "{commit_id}";')
for commit_id, second_parent in merge_parents.items():
    dot_lines.append(f'  "{second_parent}" -> "{commit_id}" [style=dashed];')

# раскрасим коммиты, на которые указывают ветки
branch_colors = {}
palette = ["orange", "lightgreen", "lightpink", "khaki", "violet", "lightcoral"]
for i, (branch_name, head_commit) in enumerate(st.session_state.branches.items()):
    color = palette[i % len(palette)]
    branch_colors[head_commit] = (branch_name, color)

for commit_id in st.session_state.commits:
    if commit_id in branch_colors:
        bname, color = branch_colors[commit_id]
        head_marker = " [HEAD]" if bname == st.session_state.current_branch else ""
        dot_lines.append(
            f'  "{commit_id}" [fillcolor="{color}", label="{commit_id}\\n{bname}{head_marker}"];'
        )

dot_lines.append("}")
dot_source = "\n".join(dot_lines)

st.graphviz_chart(dot_source, use_container_width=True)

with st.expander("Журнал действий"):
    for entry in reversed(st.session_state.log[-25:]):
        st.markdown(f"- {entry}")

st.markdown(
    """
**Как читать граф:** каждый узел — коммит, стрелки идут от родителя к потомку
(слева направо). Цветные узлы — текущие положения веток (`[HEAD]` отмечает
ветку, на которой вы сейчас находитесь). Пунктирная стрелка — второй родитель
merge-коммита (результат `git merge` двух разошедшихся веток). Если при слиянии
ветка-цель является прямым предком ветки-источника, Git делает **fast-forward**
— просто передвигает указатель ветки без создания нового коммита.
"""
)

# ---------------------------------------------------------------------------
# 2. Тренажёр команд Git
# ---------------------------------------------------------------------------

st.header("2. Тренажёр: какая команда нужна?")

st.markdown(
    "Для каждого сценария выберите команду Git, которая решает описанную задачу. "
    "Сценарии характерны для повседневной работы в ML-проекте."
)

SCENARIOS = [
    {
        "question": "Вы только начинаете новый проект и хотите создать Git-репозиторий в текущей папке.",
        "options": ["git init", "git clone .", "git start", "git new repo"],
        "answer": "git init",
        "explanation": "`git init` инициализирует новый репозиторий, создавая скрытую папку `.git` с историей.",
    },
    {
        "question": "Коллега выложил ML-проект на GitHub, вам нужна полная локальная копия со всей историей.",
        "options": ["git pull <url>", "git clone <url>", "git fetch <url>", "git fork <url>"],
        "answer": "git clone <url>",
        "explanation": "`git clone` скачивает удалённый репозиторий целиком, включая всю историю коммитов, и настраивает remote `origin`.",
    },
    {
        "question": "Вы изменили файл preprocessing.py и хотите включить эти изменения в следующий коммит.",
        "options": ["git commit preprocessing.py", "git stage preprocessing.py", "git add preprocessing.py", "git push preprocessing.py"],
        "answer": "git add preprocessing.py",
        "explanation": "`git add` переносит изменения файла из рабочей директории в staging area (индекс) — подготовку к коммиту.",
    },
    {
        "question": "Вы хотите начать работу над новой фичей (валидация данных) в отдельной ветке, созданной от текущего main.",
        "options": ["git branch feature/data-validation", "git switch -c feature/data-validation", "git merge feature/data-validation", "git checkout main feature/data-validation"],
        "answer": "git switch -c feature/data-validation",
        "explanation": "`git switch -c <branch>` (или старый вариант `git checkout -b <branch>`) создаёт новую ветку от текущего HEAD и сразу переключается на неё.",
    },
    {
        "question": "Вам нужно посмотреть построчные различия между текущей рабочей версией файла и последним коммитом, не коммитя их.",
        "options": ["git log", "git diff", "git status", "git show"],
        "answer": "git diff",
        "explanation": "`git diff` показывает построчные изменения, не внесённые в staging area / коммит.",
    },
    {
        "question": "Вы закончили работу над веткой feature/new-model и хотите отправить её на GitHub впервые.",
        "options": ["git push origin feature/new-model", "git pull origin feature/new-model", "git commit -m push", "git remote add feature/new-model"],
        "answer": "git push origin feature/new-model",
        "explanation": "`git push -u origin <branch>` (первый раз с флагом `-u` для отслеживания) отправляет локальную ветку в удалённый репозиторий `origin`.",
    },
    {
        "question": "Перед тем как открыть Pull Request, вы хотите подтянуть свежие изменения из main в свою feature-ветку, чтобы избежать больших конфликтов.",
        "options": ["git push origin main", "git clone main", "git pull origin main (или git rebase main)", "git reset --hard main"],
        "answer": "git pull origin main (или git rebase main)",
        "explanation": "Регулярная синхронизация feature-ветки с main через `pull`/`rebase` уменьшает накопление расходящихся изменений и риск крупных конфликтов.",
    },
    {
        "question": "В общей ветке main был сделан плохой коммит с багом в пайплайне обучения, который уже увидели другие. Нужно безопасно его отменить.",
        "options": ["git reset --hard HEAD~1", "git push --force", "git revert <commit>", "git rebase -i HEAD~1"],
        "answer": "git revert <commit>",
        "explanation": "`git revert` создаёт новый коммит, отменяющий изменения указанного коммита, не переписывая опубликованную историю — безопасно для общих веток.",
    },
    {
        "question": "Вы случайно начали менять файлы прямо в main, но хотите переключиться на другую ветку без коммита и без потери изменений.",
        "options": ["git stash", "git reset --hard", "git clean -fd", "git branch -D main"],
        "answer": "git stash",
        "explanation": "`git stash` временно сохраняет незакоммиченные изменения, позволяя переключиться на другую ветку и затем вернуть их через `git stash pop`.",
    },
    {
        "question": "Вы хотите узнать, изменилось ли что-то на сервере (origin), но пока не применять эти изменения к своей рабочей ветке.",
        "options": ["git pull", "git fetch", "git merge origin", "git status --remote"],
        "answer": "git fetch",
        "explanation": "`git fetch` скачивает данные из remote, но не объединяет их с текущей рабочей веткой — `git pull` делает `fetch` + `merge` за один шаг.",
    },
]

if "quiz_order" not in st.session_state:
    order = list(range(len(SCENARIOS)))
    random.shuffle(order)
    st.session_state.quiz_order = order
    st.session_state.quiz_idx = 0
    st.session_state.quiz_score = 0
    st.session_state.quiz_answered = False

idx = st.session_state.quiz_order[st.session_state.quiz_idx]
scenario = SCENARIOS[idx]

st.markdown(f"**Вопрос {st.session_state.quiz_idx + 1} из {len(SCENARIOS)}**")
st.markdown(f"> {scenario['question']}")

choice = st.radio("Выберите команду:", scenario["options"], key=f"choice_{idx}", index=None)

col_check, col_next = st.columns(2)
with col_check:
    if st.button("Проверить ответ", disabled=st.session_state.quiz_answered):
        st.session_state.quiz_answered = True
        if choice == scenario["answer"]:
            st.session_state.quiz_score += 1
            st.session_state.quiz_correct_this = True
        else:
            st.session_state.quiz_correct_this = False

if st.session_state.quiz_answered:
    if st.session_state.get("quiz_correct_this"):
        st.success(f"Верно! {scenario['explanation']}")
    else:
        st.error(f"Неверно. Правильный ответ: `{scenario['answer']}`. {scenario['explanation']}")

with col_next:
    if st.button("Следующий вопрос", disabled=not st.session_state.quiz_answered):
        st.session_state.quiz_idx += 1
        st.session_state.quiz_answered = False
        if st.session_state.quiz_idx >= len(SCENARIOS):
            st.session_state.quiz_idx = 0
            order = list(range(len(SCENARIOS)))
            random.shuffle(order)
            st.session_state.quiz_order = order
        st.rerun()

st.markdown(f"**Текущий счёт: {st.session_state.quiz_score} из {len(SCENARIOS)} (за этот проход)**")

if st.button("Сбросить тренажёр"):
    for key in ["quiz_order", "quiz_idx", "quiz_score", "quiz_answered", "quiz_correct_this"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# ---------------------------------------------------------------------------
# 3. Типичный PR-workflow
# ---------------------------------------------------------------------------

st.header("3. Типичный workflow Pull Request")

st.graphviz_chart(
    r"""
    digraph PR {
        rankdir=LR;
        node [shape=box, style="rounded,filled", fillcolor=lightyellow, fontsize=11];

        A [label="1. git switch -c feature/x\n(ветка от main)"];
        B [label="2. изменения +\ngit add / git commit"];
        C [label="3. git push -u origin\nfeature/x"];
        D [label="4. Открыть Pull Request\nна GitHub"];
        E [label="5. CI: тесты, линтеры\n(GitHub Actions)", fillcolor=lightblue];
        F [label="6. Code review:\nкомментарии, правки"];
        G [label="7. Approve"];
        H [label="8. Merge в main\n(merge / squash / rebase)", fillcolor=lightgreen];
        I [label="9. Удалить ветку\nfeature/x"];

        A -> B -> C -> D -> E;
        E -> F [label="CI прошёл"];
        E -> B [label="CI упал", style=dashed, color=red];
        F -> G [label="изменения одобрены"];
        F -> B [label="запрошены правки", style=dashed, color=red];
        G -> H -> I;
    }
    """,
    use_container_width=True,
)

st.markdown(
    """
В ML-проектах шаг **CI (GitHub Actions)** часто включает не только тесты и
линтеры, но и быстрые "смоук"-тесты пайплайна на маленьком сэмпле данных, а шаг
**code review** дополнительно проверяет воспроизводимость (зафиксированы ли seed
и версии данных) и отсутствие утечек таргета в коде препроцессинга.
"""
)
