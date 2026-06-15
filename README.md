# ds4me

Сайт-наставник для теоретической прокачки по Data Science.

## Структура

- `pages/index.html` — центральная страница со ссылками на все тематические страницы.
- `pages/ml-theory.html` — 1. Общая теория Machine Learning.
- `pages/deep-learning.html` — 1.1 Deep Learning — общая теория.
- `pages/mlops.html` — 2. MLOps.
- `pages/data-product.html` — 3. Data Product.
- `pages/ml-architect.html` — 4. ML Architect.
- `pages/experiment-design.html` — 5. Дизайн экспериментов.
- `pages/leetcode.html` — 6. LeetCode (задачи по алгоритмам и структурам данных).
- `pages/ml-system-design-interview.html` — 7. ML System Design (кейсы для
  интервью по проектированию ML-систем).
- `pages/assets/` — общие стили (`style.css`), скрипт аккордеона (`accordion.js`),
  шаблон темы (`topic-template.html`) и шаблон задачи/кейса
  (`problem-template.html`, для страниц LeetCode и ML System Design — пара
  независимых аккордеонов "Задание"/"Решение").
- `streamlit_app/` — интерактивные симуляции на Streamlit, дополняющие
  теоретические страницы. Каждая тема из `pages/*.html` имеет свою страницу
  в `streamlit_app/topics/`, подключённую через навигацию в `app.py`.
- `.claude/agents/` — определения агентов-наставников. Каждый агент сам
  предлагает темы по своей области и, после согласия пользователя,
  добавляет теорию, формулы (KaTeX) и подборку мануалов в виде новой
  секции аккордеона на свою страницу, а также (если применимо)
  интерактивную симуляцию в `streamlit_app/`.

## Как пользоваться

Откройте `pages/index.html` в браузере и переходите на нужную тематическую
страницу. Чтобы получить новую тему, запустите соответствующего агента
(`ml-theory-agent`, `deep-learning-agent`, `mlops-agent`,
`data-product-agent`, `ml-architect-agent`, `experiment-design-agent`,
`leetcode-agent`, `ml-system-design-interview-agent`) — он предложит варианты
тем/задач и после вашего согласия добавит материал на страницу и симуляцию в
Streamlit-приложение.

## Streamlit-приложение с симуляциями

Развёрнутая версия: https://4hkhqzmf4hfys8fidqxhcv.streamlit.app/ (ссылка также
есть на главной странице сайта).

Интерактивные симуляции можно запустить и локально, отдельно от статического сайта:

```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

Приложение откроется в браузере с навигацией по темам слева — каждая
страница соответствует разделу аккордеона на одной из страниц `pages/*.html`.