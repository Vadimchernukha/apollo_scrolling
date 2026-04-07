# Apollo Lead Scoring

B2B ICP-скоринг компаний из выгрузки Apollo с помощью Claude Haiku 4.5.

## Как работает

Каскадный анализ для каждой компании:
1. **Apollo данные** → Claude (Short Description, Technologies, Keywords)
2. **Парсинг сайта** → crawl4ai → Claude (если шаг 1 дал MAYBE)
3. **DuckDuckGo поиск** → Claude (если сайт недоступен или снова MAYBE)

Результат: Excel-файл с колонками `ICP_Status` (YES/NO/MAYBE/MANUAL_REVIEW), `Reason`, `Data_Source` и цветовой подсветкой.

## Локальный запуск

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# отредактируй secrets.toml: вставь ANTHROPIC_API_KEY и при желании смени пароль
streamlit run app.py
```

## Деплой на Streamlit Community Cloud

1. Залить репозиторий на GitHub (`.streamlit/secrets.toml` в `.gitignore` — не попадёт)
2. На [share.streamlit.io](https://share.streamlit.io) → New app → выбрать репо → `app.py`
3. В разделе **Secrets** вставить содержимое `secrets.toml`:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
COOKIE_SECRET = "любая-случайная-строка"

[credentials.usernames.admin]
name = "Admin"
email = ""
password = "ваш_пароль"
```

## Структура файлов

```
app.py               — Streamlit UI + авторизация + управление заданиями
scoring_logic.py     — Каскадная логика (Apollo → сайт → DDG → Claude)
profiles.yaml        — ICP-профили (редактируется вручную)
requirements.txt     — Зависимости
.streamlit/
  secrets.toml       — Секреты (gitignored)
  secrets.toml.example — Шаблон
```

## Логин по умолчанию

`admin` / `admin` (меняется в `secrets.toml`)

Сессия сохраняется в cookie на 30 дней — повторный логин не требуется.
