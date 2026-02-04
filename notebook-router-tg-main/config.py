"""
Configuration for NotebookLM Telegram Bot

100% Gemini-powered. No OpenAI dependency.
Uses Gemini 2.0 Flash with thinking capabilities.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Telegram Bot Token (get from @BotFather)
BOT_TOKEN = os.getenv("BOT_TOKEN")

# Gemini API Key - единственный API ключ для всего
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Admin user ID - only admin can add/remove stores
ADMIN_USER_ID = int(os.getenv("ADMIN_USER_ID", "0")) if os.getenv("ADMIN_USER_ID") else None

# Stores JSON file path
STORES_FILE = Path(__file__).parent / "stores.json"

# User memory file path
MEMORY_FILE = Path(__file__).parent / "user_memory.json"

# Memory settings
MEMORY_MAX_MESSAGES = int(os.getenv("MEMORY_MAX_MESSAGES", "5"))
MEMORY_CLEANUP_DAYS = int(os.getenv("MEMORY_CLEANUP_DAYS", "7"))

# Allowed Telegram user IDs (optional security)
# Leave empty to allow all users
ALLOWED_USERS = [
    int(uid.strip())
    for uid in os.getenv("ALLOWED_USERS", "").split(",")
    if uid.strip()
]

# Timeouts
QUERY_TIMEOUT = int(os.getenv("QUERY_TIMEOUT", "60"))

# Gemini 3 models
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
GEMINI_THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "medium")  # minimal, low, medium, high

# Google Drive Service Account (for /uploadurl command)
GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "service_account.json")

# Default prompt for image recognition
IMAGE_DEFAULT_PROMPT = os.getenv(
    "IMAGE_DEFAULT_PROMPT",
    "Внимательно рассмотри изображение. Если это доска/документ с текстом - распознай и перечисли всё что написано. Если это схема/диаграмма - опиши её структуру."
)
