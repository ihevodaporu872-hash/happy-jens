# NotebookLM Telegram Bot

Telegram бот для запросов к Google NotebookLM. Отправляйте вопросы в Telegram — получайте ответы из ваших документов.

## Требования

- Python 3.8+
- Настроенный [notebooklm-skill](https://github.com/PleasePrompto/notebooklm-skill) с авторизацией
- Telegram Bot Token от [@BotFather](https://t.me/BotFather)

## Установка

```bash
# 1. Клонируйте репозиторий
git clone https://github.com/PleasePrompto/notebooklm-telegram-bot
cd notebooklm-telegram-bot

# 2. Создайте виртуальное окружение
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Установите зависимости
pip install -r requirements.txt

# 4. Создайте .env файл
copy .env.example .env  # Windows
# cp .env.example .env  # Linux/Mac

# 5. Отредактируйте .env - добавьте BOT_TOKEN
```

## Настройка

### Получение Bot Token

1. Откройте [@BotFather](https://t.me/BotFather) в Telegram
2. Отправьте `/newbot`
3. Следуйте инструкциям
4. Скопируйте токен в `.env`

### Настройка notebooklm-skill

Убедитесь, что [notebooklm-skill](https://github.com/PleasePrompto/notebooklm-skill) настроен:

```bash
cd ../notebooklm-skill
.venv\Scripts\python scripts\auth_manager.py status
```

Если не авторизован — запустите `auth_manager.py setup`.

## Запуск

```bash
python bot.py
```

## Команды бота

| Команда | Описание |
|---------|----------|
| `/start` | Приветствие и справка |
| `/set <url>` | Установить активный блокнот |
| `/list` | Показать сохранённые блокноты |
| `/status` | Проверить статус авторизации |
| *любой текст* | Запрос к NotebookLM |

## Пример использования

```
Вы: /set https://notebooklm.google.com/notebook/abc123
Бот: Notebook set!

Вы: Какие требования к API авторизации?
Бот: Согласно документации, API требует OAuth 2.0...
```

## Безопасность

Для ограничения доступа добавьте в `.env`:

```
ALLOWED_USERS=123456789,987654321
```

Узнать свой Telegram ID можно через [@userinfobot](https://t.me/userinfobot).

## Структура проекта

```
notebooklm-telegram-bot/
├── bot.py              # Основной код бота
├── config.py           # Конфигурация
├── requirements.txt    # Зависимости
├── .env.example        # Пример настроек
└── .gitignore
```

## Связанные проекты

- [notebooklm-skill](https://github.com/PleasePrompto/notebooklm-skill) — Claude Code Skill для NotebookLM
- [notebooklm-mcp](https://github.com/PleasePrompto/notebooklm-mcp) — MCP сервер для NotebookLM
