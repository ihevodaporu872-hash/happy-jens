"""
Configuration for NotebookLM Telegram Bot
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Telegram Bot Token (get from @BotFather)
BOT_TOKEN = os.getenv("BOT_TOKEN")

# OpenAI API Key for smart routing
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Alias for compatibility
ANTHROPIC_API_KEY = OPENAI_API_KEY  # Using OpenAI instead

# Admin user ID - only admin can add/remove notebooks
ADMIN_USER_ID = int(os.getenv("ADMIN_USER_ID", "0")) if os.getenv("ADMIN_USER_ID") else None

# Path to NotebookLM skill (inside project directory)
SKILL_PATH = Path(os.getenv(
    "NOTEBOOKLM_SKILL_PATH",
    Path(__file__).parent / "notebooklm-skill"  # Fixed: was parent.parent
))

# Scripts path
SCRIPTS_PATH = SKILL_PATH / "scripts"

# Python executable in skill's venv
SKILL_PYTHON = SKILL_PATH / ".venv" / "Scripts" / "python.exe"

# Notebooks JSON file path
NOTEBOOKS_FILE = Path(__file__).parent / "notebooks.json"

# Allowed Telegram user IDs (optional security)
# Leave empty to allow all users
ALLOWED_USERS = [
    int(uid.strip())
    for uid in os.getenv("ALLOWED_USERS", "").split(",")
    if uid.strip()
]

# Default notebook URL (optional)
DEFAULT_NOTEBOOK_URL = os.getenv("DEFAULT_NOTEBOOK_URL", "")

# Timeouts
QUERY_TIMEOUT = int(os.getenv("QUERY_TIMEOUT", "180"))  # seconds

# Max parallel notebook queries
MAX_PARALLEL_QUERIES = int(os.getenv("MAX_PARALLEL_QUERIES", "3"))

# Auto sync interval in minutes (0 to disable)
# Recommended: 30 minutes for safety
AUTO_SYNC_INTERVAL = int(os.getenv("AUTO_SYNC_INTERVAL", "30"))
