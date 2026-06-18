import random
import subprocess

import streamlit as st

st.title("7. Командная строка (Linux/Unix) для ML-инженера")
st.caption("Теория — pages/mlops.html, раздел 7.")

st.markdown(
    r"""
Командная строка — основной инструмент для работы с удалёнными серверами,
контейнерами и облачными GPU-машинами. На macOS многие команды совпадают с
Linux (обе системы — Unix), но есть систематические отличия: оболочка по
умолчанию — `zsh` (а не `bash`), а базовые утилиты — BSD-вариант (а не GNU,
как почти везде на Linux).

Эта страница — две интерактивные части:

1. **Живая демонстрация** — реальный вывод нескольких безопасных
   read-only команд, выполненных прямо в этом окружении.
2. **Тренажёр команд** — типичные сценарии, нужно ввести команду, которая
   решает задачу, и проверить её против списка допустимых ответов.
"""
)

# ---------------------------------------------------------------------------
# 1. Живая демонстрация (whitelist безопасных read-only команд)
# ---------------------------------------------------------------------------

st.header("1. Живая демонстрация: реальный вывод команд")

st.markdown(
    "Из соображений безопасности (это приложение может работать в общем "
    "облачном окружении) здесь выполняется **только фиксированный список "
    "безопасных read-only команд** — никакой произвольный пользовательский "
    "ввод никогда не передаётся в shell. Выберите команду из списка и "
    "посмотрите на настоящий вывод."
)

# Жёсткий whitelist: команда -> (список argv без shell=True, описание)
SAFE_COMMANDS = {
    "pwd — текущая директория": ["pwd"],
    "ls -la /tmp — список файлов /tmp": ["ls", "-la", "/tmp"],
    "df -h — использование дискового пространства": ["df", "-h"],
    "uname -a — информация об ОС и ядре": ["uname", "-a"],
    "ps aux (первые 10 строк) — запущенные процессы": ["ps", "aux"],
    "whoami — текущий пользователь": ["whoami"],
    "date — текущая дата и время": ["date"],
    "echo $PATH (через env) — переменные окружения": ["env"],
}

choice_label = st.selectbox("Выберите команду для запуска:", list(SAFE_COMMANDS.keys()))
argv = SAFE_COMMANDS[choice_label]

st.code("$ " + " ".join(argv), language="bash")

if st.button("Выполнить команду"):
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=5,
            shell=False,  # критично: всегда список argv, никогда строка/shell=True
        )
        output = result.stdout or ""
        # для длинных выводов (ls, ps, env) показываем только первые строки
        lines = output.splitlines()
        if len(lines) > 15:
            output = "\n".join(lines[:15]) + f"\n... (показаны первые 15 из {len(lines)} строк)"
        st.code(output if output.strip() else "(пустой вывод)", language="text")
        if result.stderr:
            st.caption(f"stderr: {result.stderr.strip()[:300]}")
    except FileNotFoundError:
        st.error("Команда не найдена в этом окружении (возможно, не установлена).")
    except subprocess.TimeoutExpired:
        st.error("Команда не завершилась за отведённое время.")

st.markdown(
    """
**Почему так строго.** В реальном терминале вы можете ввести что угодно. Но
веб-приложение, выполняющее произвольный пользовательский текст в shell
(`subprocess.run(user_input, shell=True)`), — это классическая уязвимость
command injection. Поэтому здесь пользователь только **выбирает** одну из
заранее зашитых в код безопасных команд — на сервер не уходит ни одного
символа произвольного ввода.
"""
)

# ---------------------------------------------------------------------------
# 2. Тренажёр команд
# ---------------------------------------------------------------------------

st.header("2. Тренажёр: какая команда нужна?")

st.markdown(
    "Для каждого сценария введите команду, которая решает описанную задачу. "
    "Проверка нестрогая: достаточно совпадения с одним из принятых вариантов "
    "(допускаются небольшие отличия в пробелах)."
)

SCENARIOS = [
    {
        "question": "Найти все файлы с расширением .log, изменённые за последние 24 часа, в текущей директории и поддиректориях.",
        "answers": [
            "find . -name *.log -mtime -1",
            "find . -name \"*.log\" -mtime -1",
            "find . -iname *.log -mtime -1",
        ],
        "explanation": "`find . -name \"*.log\" -mtime -1` ищет от текущей директории (`.`) файлы по паттерну имени, изменённые менее 1 дня назад (`-mtime -1`).",
    },
    {
        "question": "Показать 5 самых больших файлов/директорий в текущей директории, отсортированных по размеру (от большего к меньшему).",
        "answers": [
            "du -sh * | sort -rh | head -5",
            "du -sh * | sort -rh | head -n 5",
            "du -h * | sort -rh | head -5",
        ],
        "explanation": "`du -sh *` считает размер каждого файла/директории в человекочитаемом виде, `sort -rh` сортирует по размеру в обратном порядке, `head -5` оставляет первые 5 строк.",
    },
    {
        "question": "Найти PID процесса по имени (например, всех процессов python) и завершить его вежливо (SIGTERM).",
        "answers": [
            "kill $(pgrep python)",
            "pkill python",
            "kill `pgrep python`",
            "ps aux | grep python",
        ],
        "explanation": "`pkill python` находит процессы по имени и отправляет им SIGTERM; альтернативно — найти PID через `pgrep python` и передать в `kill`. `kill -9` (SIGKILL) нужно использовать только если процесс не реагирует на обычный kill.",
    },
    {
        "question": "Посмотреть последние 50 строк лог-файла train.log и продолжать следить за новыми строками по мере их появления.",
        "answers": [
            "tail -n 50 -f train.log",
            "tail -f -n 50 train.log",
            "tail -50 -f train.log",
            "tail -f train.log",
        ],
        "explanation": "`tail -f` переключает в режим непрерывного отслеживания файла (follow), флаг `-n 50` задаёт, сколько последних строк показать перед началом отслеживания.",
    },
    {
        "question": "В файле service.log посчитать, сколько строк содержат слово 'ERROR' (без учёта регистра).",
        "answers": [
            "grep -ic error service.log",
            "grep -i error service.log | wc -l",
            "grep -ci error service.log",
        ],
        "explanation": "`grep -i` ищет без учёта регистра, `-c` считает число совпадающих строк (либо то же самое через пайп с `wc -l`).",
    },
    {
        "question": "Установить пакет ffmpeg на Mac, следуя Linux-инструкции, в которой написано 'sudo apt-get install ffmpeg'.",
        "answers": [
            "brew install ffmpeg",
        ],
        "explanation": "На macOS системного apt/yum нет — стандартный аналог это Homebrew: `brew install ffmpeg`.",
    },
    {
        "question": "Сделать файл deploy.sh исполняемым.",
        "answers": [
            "chmod +x deploy.sh",
            "chmod 755 deploy.sh",
            "chmod 744 deploy.sh",
        ],
        "explanation": "`chmod +x` добавляет право на выполнение; числовой эквивалент с правами rwx для владельца — `chmod 755` или `744`.",
    },
    {
        "question": "Запустить длительное обучение python train.py так, чтобы оно не прервалось при закрытии SSH-сессии, и продолжить работу с терминалом.",
        "answers": [
            "nohup python train.py &",
            "nohup python train.py > train.log &",
        ],
        "explanation": "`nohup ... &` запускает процесс в фоне и защищает его от сигнала разрыва соединения (hangup) при закрытии терминала/SSH.",
    },
    {
        "question": "Скопировать локальный файл model.pt на удалённый сервер по SSH в директорию /models/.",
        "answers": [
            "scp model.pt user@host:/models/",
            "scp model.pt user@host:/models",
        ],
        "explanation": "`scp <локальный файл> user@host:<путь>` копирует файл на удалённую машину по SSH-протоколу.",
    },
    {
        "question": "Проверить, что REST-эндпоинт модели http://localhost:8000/health отвечает (быстрая проверка из терминала).",
        "answers": [
            "curl http://localhost:8000/health",
            "curl -X GET http://localhost:8000/health",
        ],
        "explanation": "`curl <url>` по умолчанию делает GET-запрос и выводит тело ответа — базовый способ быстро проверить доступность HTTP-эндпоинта.",
    },
]


def normalize(s: str) -> str:
    return " ".join(s.strip().split())


if "cli_quiz_order" not in st.session_state:
    order = list(range(len(SCENARIOS)))
    random.shuffle(order)
    st.session_state.cli_quiz_order = order
    st.session_state.cli_quiz_idx = 0
    st.session_state.cli_quiz_score = 0
    st.session_state.cli_quiz_checked = False

idx = st.session_state.cli_quiz_order[st.session_state.cli_quiz_idx]
scenario = SCENARIOS[idx]

st.markdown(f"**Вопрос {st.session_state.cli_quiz_idx + 1} из {len(SCENARIOS)}**")
st.markdown(f"> {scenario['question']}")

user_cmd = st.text_input("Введите команду:", key=f"cli_input_{idx}")

col_check, col_next = st.columns(2)
with col_check:
    if st.button("Проверить", key=f"check_{idx}"):
        st.session_state.cli_quiz_checked = True
        is_correct = normalize(user_cmd) in [normalize(a) for a in scenario["answers"]]
        st.session_state.cli_quiz_correct_this = is_correct
        if is_correct:
            st.session_state.cli_quiz_score += 1

if st.session_state.cli_quiz_checked:
    accepted = " | ".join(f"`{a}`" for a in scenario["answers"])
    if st.session_state.get("cli_quiz_correct_this"):
        st.success(f"Верно! {scenario['explanation']}")
    else:
        st.error(f"Не совпало с принятыми вариантами: {accepted}\n\n{scenario['explanation']}")

with col_next:
    if st.button("Следующий вопрос", disabled=not st.session_state.cli_quiz_checked):
        st.session_state.cli_quiz_idx += 1
        st.session_state.cli_quiz_checked = False
        if st.session_state.cli_quiz_idx >= len(SCENARIOS):
            st.session_state.cli_quiz_idx = 0
            order = list(range(len(SCENARIOS)))
            random.shuffle(order)
            st.session_state.cli_quiz_order = order
        st.rerun()

st.markdown(f"**Текущий счёт: {st.session_state.cli_quiz_score} из {len(SCENARIOS)} (за этот проход)**")

if st.button("Сбросить тренажёр"):
    for key in [
        "cli_quiz_order",
        "cli_quiz_idx",
        "cli_quiz_score",
        "cli_quiz_checked",
        "cli_quiz_correct_this",
    ]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# ---------------------------------------------------------------------------
# 3. Шпаргалка macOS vs Linux
# ---------------------------------------------------------------------------

st.header("3. Шпаргалка: macOS (BSD) vs Linux (GNU)")

st.markdown(
    """
| Команда / тема | Linux (GNU) | macOS (BSD) по умолчанию | GNU-поведение на Mac |
|---|---|---|---|
| `sed -i` | `sed -i 's/a/b/' file` | `sed -i '' 's/a/b/' file` (нужен явный аргумент) | `brew install gnu-sed` → `gsed` |
| `date` | `date -d "yesterday"` | `date -v-1d` | `brew install coreutils` → `gdate -d ...` |
| `find` | поддерживает `-printf` | нет `-printf` | `brew install findutils` → `gfind` |
| `ls` цвета | `ls --color=auto` | `ls -G` | `brew install coreutils` → `gls --color=auto` |
| Менеджер пакетов | `apt-get` / `yum` | нет системного — ставится **Homebrew** | `brew install <пакет>` |
| Shell по умолчанию | `bash` | `zsh` (с Catalina) | `chsh -s /bin/bash` (не обязательно) |
| Файл профиля | `~/.bashrc` | `~/.zshrc` | — |
"""
)
