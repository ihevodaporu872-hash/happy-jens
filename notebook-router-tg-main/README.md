# Tender Document Telegram Bot (Gemini)

Telegram бот для работы с тендерной документацией через Gemini.
Каждый **store** = отдельный тендер с загруженными документами (PDF, DOCX и т.д.).

## Требования

- Python 3.10+
- Telegram Bot Token от @BotFather
- Gemini API Key (Google AI Studio)
- (Опционально) Google Drive Service Account для доступа к папкам

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
copy .env.example .env  # Windows
# cp .env.example .env  # Linux/Mac
```

Заполни `.env` как минимум `BOT_TOKEN` и `GEMINI_API_KEY`.

## Запуск

```bash
python bot.py
```

## Команды

| Команда | Описание |
|---------|----------|
| `/start`, `/help` | Приветствие и справка |
| `/list`, `/stores` | Список всех stores |
| `/select <store>` | Выбрать активный store |
| `/status` | Статус и конфигурация |
| `/clear` | Очистить историю диалога |
| `/compare <s1> <s2> <topic>` | Сравнить два тендера |
| `/export` | Экспорт последнего ответа в PDF/DOCX |
| `/add`, `/addstore` | Создать store (админ) |
| `/delete`, `/deletestore` | Удалить store (админ) |
| `/rename <old> | <new>` | Переименовать store (админ) |
| `/upload` | Загрузить файл в store (админ) |
| `/uploadurl` | Загрузить из Google URL (админ) |
| `/setsync` | Настроить автосинк (админ) |
| `/syncnow` | Принудительный синк (админ) |

## Натуральный язык (без команд)

Бот понимает запросы вида:
- "Покажи список тендеров"
- "Выбери тендер Дубровка"
- "Сделай экспорт в PDF"
- "Сравни Дубровка и МайПриорити по земляным работам"

## Структура проекта

```
notebook-router-tg-main/
├── bot.py              # Основной код бота
├── config.py           # Конфигурация
├── gemini_client.py    # Gemini File Search
├── query_processor.py  # AI понимание запросов
├── router.py           # Роутинг по stores
├── memory_client.py    # Память пользователя
├── export_client.py    # Экспорт PDF/DOCX
├── google_drive_client.py # Загрузка из Google Drive
└── requirements.txt
```
