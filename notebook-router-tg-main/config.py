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

# Optional notification channel/chat ID
NOTIFICATION_CHANNEL_ID = (
    int(os.getenv("NOTIFICATION_CHANNEL_ID", "0"))
    if os.getenv("NOTIFICATION_CHANNEL_ID")
    else None
)

# Stores JSON file path
STORES_FILE = Path(__file__).parent / "stores.json"

# User memory file path
MEMORY_FILE = Path(__file__).parent / "user_memory.json"

# User state file path (selected store per user)
USER_STATE_FILE = Path(__file__).parent / "user_state.json"

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

# Confidence threshold for command-like intent actions
ACTION_CONFIDENCE_THRESHOLD = float(os.getenv("ACTION_CONFIDENCE_THRESHOLD", "0.6"))

# Gemini 3 models
# Flash - for simple/medium tasks (fast, efficient)
GEMINI_MODEL_FLASH = os.getenv("GEMINI_MODEL_FLASH", "gemini-3-flash-preview")
# Pro - for complex tasks (query understanding, comparisons, deep analysis)
GEMINI_MODEL_PRO = os.getenv("GEMINI_MODEL_PRO", "gemini-3-pro-preview")

# Default model (backwards compatibility)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", GEMINI_MODEL_FLASH)

# Google Drive Service Account (for /uploadurl command)
GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "service_account.json")

# Default prompt for image recognition
IMAGE_DEFAULT_PROMPT = os.getenv(
    "IMAGE_DEFAULT_PROMPT",
    "Внимательно рассмотри изображение. Если это доска/документ с текстом - распознай и перечисли всё что написано. Если это схема/диаграмма - опиши её структуру."
)
