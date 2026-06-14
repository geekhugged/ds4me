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
`data-product-agent`, `ml-architect-agent`) — он предложит варианты тем и
после вашего согласия добавит материал на страницу и симуляцию в
Streamlit-приложение.

## Streamlit-приложение с симуляциями

Интерактивные симуляции запускаются отдельно от статического сайта:

```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

Приложение откроется в браузере с навигацией по темам слева — каждая
страница соответствует разделу аккордеона на одной из страниц `pages/*.html`.