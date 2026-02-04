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
