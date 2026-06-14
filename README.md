# ds4me

Сайт-наставник для теоретической прокачки по Data Science.

## Структура

- `pages/index.html` — центральная страница со ссылками на все тематические страницы.
- `pages/ml-theory.html` — 1. Общая теория Machine Learning.
- `pages/deep-learning.html` — 1.1 Deep Learning — общая теория.
- `pages/mlops.html` — 2. MLOps.
- `pages/data-product.html` — 3. Data Product.
- `pages/ml-architect.html` — 4. ML Architect.
- `pages/assets/` — общие стили (`style.css`), скрипт аккордеона (`accordion.js`)
  и шаблон темы (`topic-template.html`).
- `.claude/agents/` — определения агентов-наставников. Каждый агент сам
  предлагает темы по своей области и, после согласия пользователя,
  добавляет теорию, формулы (KaTeX) и подборку мануалов в виде новой
  секции аккордеона на свою страницу.

## Как пользоваться

Откройте `pages/index.html` в браузере и переходите на нужную тематическую
страницу. Чтобы получить новую тему, запустите соответствующего агента
(`ml-theory-agent`, `deep-learning-agent`, `mlops-agent`,
`data-product-agent`, `ml-architect-agent`) — он предложит варианты тем и
после вашего согласия добавит материал на страницу.