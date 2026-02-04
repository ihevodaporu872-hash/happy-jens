#!/usr/bin/env python3
"""
NotebookLM Telegram Bot - Gemini 3 Flash Powered

Uses Gemini 3 Flash for everything:
- File Search API for document Q&A
- Smart routing between stores
- Thinking levels (minimal/low/medium/high)

No OpenAI dependency. Pure Gemini 3.
"""

import logging
import sys
import re
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Tuple

from datetime import time, datetime

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

from config import (
    BOT_TOKEN,
    GEMINI_API_KEY,
    ALLOWED_USERS,
    QUERY_TIMEOUT,
    ADMIN_USER_ID,
    STORES_FILE,
    USER_STATE_FILE,
    GEMINI_MODEL,
    GEMINI_MODEL_FLASH,
    GEMINI_MODEL_PRO,
    ACTION_CONFIDENCE_THRESHOLD,
    GOOGLE_SERVICE_ACCOUNT_FILE,
    IMAGE_DEFAULT_PROMPT,
    MEMORY_FILE,
    MEMORY_MAX_MESSAGES,
    MEMORY_CLEANUP_DAYS,
    NOTIFICATION_CHANNEL_ID,
)
from router import NotebookRouter
from gemini_client import GeminiFileSearchClient
from query_processor import QueryProcessor
from google_drive_client import GoogleDriveClient
from memory_client import UserMemoryClient
from export_client import ExportClient
from user_state import UserStateClient
from intent_utils import infer_action_from_text, extract_target_store_hint

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Initialize Gemini client
gemini_client = None
if GEMINI_API_KEY:
    gemini_client = GeminiFileSearchClient(GEMINI_API_KEY, STORES_FILE)
    logger.info(f"Gemini File Search enabled. Stores: {len(gemini_client.stores)}")
else:
    logger.warning("GEMINI_API_KEY not set - file search disabled")

# Initialize router (now uses Gemini)
router = None
if GEMINI_API_KEY and gemini_client:
    router = NotebookRouter(GEMINI_API_KEY, STORES_FILE, model=GEMINI_MODEL)
    logger.info("Smart routing enabled with Gemini")
else:
    logger.warning("Router not initialized (missing API key)")

# Initialize Google Drive client
# Will work with Service Account for full access, or without for public URLs only
drive_client = None
if Path(GOOGLE_SERVICE_ACCOUNT_FILE).exists():
    drive_client = GoogleDriveClient(GOOGLE_SERVICE_ACCOUNT_FILE)
    if drive_client.is_configured():
        logger.info("Google Drive client initialized with Service Account")
    else:
        logger.warning("Google Drive Service Account failed, will use public URLs only")
else:
    # Create client without Service Account - will use public URLs only
    drive_client = GoogleDriveClient(GOOGLE_SERVICE_ACCOUNT_FILE)
    logger.info("Google Drive client initialized (public URLs only, no Service Account)")

# Initialize memory client
memory_client = UserMemoryClient(MEMORY_FILE, max_messages=MEMORY_MAX_MESSAGES)
logger.info(f"Memory client initialized (max {MEMORY_MAX_MESSAGES} messages per context)")

# Initialize user state client (selected store per user)
user_state = UserStateClient(USER_STATE_FILE)
logger.info("User state client initialized")

# Initialize export client
export_client = ExportClient()
logger.info("Export client initialized")

# Initialize query processor (uses Pro model for complex understanding)
query_processor = None
if GEMINI_API_KEY:
    query_processor = QueryProcessor(GEMINI_API_KEY, model=GEMINI_MODEL_PRO)
    logger.info(f"Query processor initialized with Pro model: {GEMINI_MODEL_PRO}")


def check_user_allowed(user_id: int) -> bool:
    """Check if user is allowed to use the bot"""
    if not ALLOWED_USERS:
        return True
    return user_id in ALLOWED_USERS


def is_admin(user_id: int) -> bool:
    """Check if user is the admin"""
    return ADMIN_USER_ID is not None and user_id == ADMIN_USER_ID


def _make_temp_path(prefix: str, suffix: str) -> Path:
    temp_dir = Path(tempfile.gettempdir()) / "notebook_router_bot"
    temp_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}_{uuid.uuid4().hex}{suffix}"
    return temp_dir / filename


def _resolve_store_by_name(name: str) -> Optional[dict]:
    if not gemini_client or not name:
        return None
    return gemini_client.find_store_by_name(name)


def _get_selected_store_for_user(user_id: int) -> Optional[dict]:
    if not gemini_client:
        return None
    state = user_state.get_selected_store(user_id)
    if not state:
        return None
    store_id = state.get("selected_store_id")
    if store_id:
        return gemini_client.get_store_by_id(store_id)
    return None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    if not check_user_allowed(update.effective_user.id):
        await update.message.reply_text("Access denied.")
        return

    gemini_status = "enabled" if gemini_client else "disabled (no API key)"
    routing_status = "enabled" if router else "disabled"
    admin_note = " (you are admin)" if is_admin(update.effective_user.id) else ""
    stores_count = len(gemini_client.stores) if gemini_client else 0

    if drive_client and drive_client.is_configured():
        drive_status = "enabled (Service Account)"
    else:
        drive_status = "enabled (public URLs only)"

    notification_status = "enabled" if NOTIFICATION_CHANNEL_ID else "disabled"

    await update.message.reply_text(
        f"Happy-Jens Tender Bot{admin_note}\n\n"
        f"Model: {GEMINI_MODEL}\n"
        f"File Search: {gemini_status}\n"
        f"Smart routing: {routing_status}\n"
        f"Google Drive: {drive_status}\n"
        f"Notifications: {notification_status}\n"
        f"Stores: {stores_count}\n\n"
        "📋 Анализ тендеров:\n"
        "/summary [store] - Executive Summary тендера\n"
        "/generate_rfi [store] [тема] - RFI письмо\n"
        "/norm <СП/ГОСТ> [store] - Поиск норматива\n\n"
        "📁 Управление:\n"
        "/list or /stores - Список тендеров\n"
        "/select <store> - Выбрать тендер\n"
        "/status - Статус бота\n"
        "/clear - Очистить историю\n\n"
        "🔍 Поиск и сравнение:\n"
        "/think <question> - Глубокий анализ\n"
        "/compare <s1> <s2> <topic> - Сравнить\n"
        "/export - Экспорт в PDF/DOCX\n\n"
        "⚙️ Админ:\n"
        "/add, /delete, /rename - Управление stores\n"
        "/upload, /uploadurl - Загрузка файлов\n"
        "/setsync, /syncnow - Синхронизация\n\n"
        "Также можно:\n"
        "• Отправить текст для поиска\n"
        "• Ссылку на Google Drive папку\n"
        "• Фото или голосовое\n\n"
        "Бот помнит последние 5 сообщений."
    )


async def add_store(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /add command - create new store (admin only)"""
    user_id = update.effective_user.id

    if not check_user_allowed(user_id):
        await update.message.reply_text("Access denied.")
        return

    if not is_admin(user_id):
        await update.message.reply_text("Only admin can add stores.")
        return

    if not gemini_client:
        await update.message.reply_text("Gemini API not configured.")
        return

    # Parse: /add <name> | <description>
    message_text = update.message.text
    args_text = re.sub(r'^/add\s*', '', message_text, flags=re.IGNORECASE).strip()

    if not args_text:
        await update.message.reply_text(
            "Usage: /add <name> | <description>\n\n"
            "Example:\n"
            "/add My Project | Documentation and guides for my project"
        )
        return

    parts = args_text.split("|", 1)
    name = parts[0].strip()
    description = parts[1].strip() if len(parts) > 1 else ""

    if not name:
        await update.message.reply_text("Please provide a store name.")
        return

    status_msg = await update.message.reply_text(f"Creating store '{name}'...")

    result = gemini_client.create_store(name, description)

    if result:
        if router:
            router.reload_library()

        await status_msg.edit_text(
            f"Store created!\n\n"
            f"Name: {result['name']}\n"
            f"ID: {result['id']}\n\n"
            f"Now upload files with:\n"
            f"/upload {name} (and attach a file)"
        )
    else:
        await status_msg.edit_text("Failed to create store. Check logs.")


async def upload_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /upload command - upload file to store (admin only)"""
    user_id = update.effective_user.id

    if not check_user_allowed(user_id):
        await update.message.reply_text("Access denied.")
        return

    if not is_admin(user_id):
        await update.message.reply_text("Only admin can upload files.")
        return

    if not gemini_client:
        await update.message.reply_text("Gemini API not configured.")
        return

    document = update.message.document
    if not document:
        args_text = re.sub(r'^/upload\s*', '', update.message.text, flags=re.IGNORECASE).strip()
        if args_text:
            context.user_data["upload_store"] = args_text
            await update.message.reply_text(
                f"Ready to upload to '{args_text}'.\n"
                f"Now send a file (PDF, TXT, DOCX, etc.)"
            )
        else:
            await update.message.reply_text(
                "Usage: /upload <store_name>\n"
                "Then send a file.\n\n"
                "Or reply to a file with /upload <store_name>"
            )
        return

    args_text = re.sub(r'^/upload\s*', '', update.message.text, flags=re.IGNORECASE).strip()
    store_name = args_text or context.user_data.get("upload_store")

    if not store_name:
        await update.message.reply_text("Please specify store name: /upload <store_name>")
        return

    store = gemini_client.find_store_by_name(store_name)
    if not store:
        await update.message.reply_text(f"Store not found: {store_name}")
        return

    status_msg = await update.message.reply_text(f"Downloading file...")

    try:
        file = await document.get_file()
        suffix = Path(document.file_name).suffix if document.file_name else ""
        temp_path = _make_temp_path("upload", suffix)
        await file.download_to_drive(temp_path)

        await status_msg.edit_text(f"Uploading to '{store_name}'...")

        success = gemini_client.upload_file(
            store["id"],
            temp_path,
            document.file_name
        )

        temp_path.unlink(missing_ok=True)
        context.user_data.pop("upload_store", None)

        if success:
            await status_msg.edit_text(
                f"File uploaded!\n\n"
                f"Store: {store_name}\n"
                f"File: {document.file_name}\n\n"
                f"You can now ask questions about this document."
            )
        else:
            await status_msg.edit_text("Failed to upload. Check logs.")

    except Exception as e:
        logger.exception("Upload error")
        await status_msg.edit_text(f"Error: {str(e)[:200]}")


async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle file uploads (for pending /upload command)"""
    user_id = update.effective_user.id

    if not check_user_allowed(user_id) or not is_admin(user_id):
        return

    store_name = context.user_data.get("upload_store")
    if not store_name:
        return

    update.message.text = f"/upload {store_name}"
    await upload_file(update, context)


async def upload_from_url(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /uploadurl command - upload files from Google URLs (admin only)"""
    user_id = update.effective_user.id

    if not check_user_allowed(user_id):
        await update.message.reply_text("Access denied.")
        return

    if not is_admin(user_id):
        await update.message.reply_text("Only admin can upload files.")
        return

    if not gemini_client:
        await update.message.reply_text("Gemini API not configured.")
        return

    # drive_client is always initialized now (with or without Service Account)

    # Parse: /uploadurl <store_name> <url1> [url2] ...
    message_text = update.message.text
    args_text = re.sub(r'^/uploadurl\s*', '', message_text, flags=re.IGNORECASE).strip()

    if not args_text:
        sa_note = ""
        if not drive_client.is_configured():
            sa_note = "\n\nNote: No Service Account configured.\nOnly public files (\"Anyone with the link\") will work.\nFolders require Service Account."

        await update.message.reply_text(
            "Usage: /uploadurl <store_name> <url1> [url2] ...\n\n"
            "Supported URLs:\n"
            "- Google Docs (exports as PDF)\n"
            "- Google Sheets (exports as XLSX)\n"
            "- Google Slides (exports as PDF)\n"
            "- Google Drive files\n"
            "- Google Drive folders (all files, requires Service Account)\n\n"
            "Example:\n"
            "/uploadurl MyStore https://docs.google.com/document/d/xxx/edit\n\n"
            f"Limits: max 10 URLs, max 50 files per folder{sa_note}"
        )
        return

    # Extract store name (first word) and URLs
    parts = args_text.split(None, 1)
    store_name = parts[0]
    urls_text = parts[1] if len(parts) > 1 else ""

    store = gemini_client.find_store_by_name(store_name)
    if not store:
        await update.message.reply_text(f"Store not found: {store_name}")
        return

    # Extract all Google URLs from the message
    urls = GoogleDriveClient.extract_all_urls(urls_text)
    if not urls:
        await update.message.reply_text(
            "No valid Google URLs found.\n"
            "Supported: docs.google.com, drive.google.com"
        )
        return

    status_msg = await update.message.reply_text(
        f"Found {len(urls)} URL(s). Processing..."
    )

    # Create temp directory
    import tempfile
    temp_dir = Path(tempfile.mkdtemp(prefix="gdrive_"))

    success_count = 0
    error_count = 0
    results = []

    try:
        for i, (url, file_id, file_type) in enumerate(urls):
            await status_msg.edit_text(
                f"Processing {i+1}/{len(urls)}: {file_type}..."
            )

            try:
                if file_type == 'folder':
                    # Folders require Service Account
                    if not drive_client.is_configured():
                        error_count += 1
                        results.append(f"- folder {file_id[:20]}... (requires Service Account)")
                        continue

                    # Download all files from folder
                    downloaded = drive_client.download_folder(file_id, temp_dir)
                    for file_path, file_name in downloaded:
                        success = gemini_client.upload_file(
                            store["id"],
                            file_path,
                            file_name
                        )
                        if success:
                            success_count += 1
                            results.append(f"+ {file_name}")
                        else:
                            error_count += 1
                            results.append(f"- {file_name} (upload failed)")
                        file_path.unlink(missing_ok=True)
                else:
                    # Download single file (pass file_type for public URL fallback)
                    file_path = drive_client.download_file(file_id, temp_dir, file_type=file_type)
                    if file_path:
                        file_name = file_path.name
                        success = gemini_client.upload_file(
                            store["id"],
                            file_path,
                            file_name,
                            source_url=url  # Save source URL for citations
                        )
                        if success:
                            success_count += 1
                            results.append(f"+ {file_name}")
                        else:
                            error_count += 1
                            results.append(f"- {file_name} (upload failed)")
                        file_path.unlink(missing_ok=True)
                    else:
                        error_count += 1
                        results.append(f"- {file_id[:20]}... (download failed)")

            except Exception as e:
                logger.error(f"Error processing {file_id}: {e}")
                error_count += 1
                results.append(f"- {file_id[:20]}... ({str(e)[:30]})")

        # Clean up temp dir
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

        # Format results
        results_text = "\n".join(results[:20])
        if len(results) > 20:
            results_text += f"\n... and {len(results) - 20} more"

        await status_msg.edit_text(
            f"Upload complete!\n\n"
            f"Store: {store_name}\n"
            f"Success: {success_count}\n"
            f"Errors: {error_count}\n\n"
            f"Files:\n{results_text}"
        )

    except Exception as e:
        logger.exception("Error in uploadurl")
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        await status_msg.edit_text(f"Error: {str(e)[:500]}")


async def list_stores(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /list command - show all stores"""
    if not check_user_allowed(update.effective_user.id):
        await update.message.reply_text("Access denied.")
        return

    if not gemini_client:
        await update.message.reply_text("Gemini API not configured.")
        return

    stores = gemini_client.list_stores()

    if not stores:
        await update.message.reply_text(
            "No stores yet.\n"
            "Admin can create with /add command."
        )
        return

    text = f"Knowledge Stores ({len(stores)}):\n\n"
    for i, store in enumerate(stores, 1):
        name = store.get("name", "Unnamed")
        desc = store.get("description", "")
        docs = len(store.get("documents", []))
        text += f"{i}. {name}\n"
        if desc:
            text += f"   {desc[:50]}{'...' if len(desc) > 50 else ''}\n"
        text += f"   Documents: {docs}\n\n"

    await update.message.reply_text(text)


async def select_store(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /select command - set active store for user"""
    user_id = update.effective_user.id

    if not check_user_allowed(user_id):
        await update.message.reply_text("Access denied.")
        return

    if not gemini_client:
        await update.message.reply_text("Gemini API not configured.")
        return

    args_text = re.sub(r'^/select\\s*', '', update.message.text, flags=re.IGNORECASE).strip()
    if not args_text:
        current = _get_selected_store_for_user(user_id)
        current_name = current.get("name") if current else "None"
        await update.message.reply_text(
            "Usage: /select <store_name>\n"
            "Example: /select Дубровка\n\n"
            f"Current selected store: {current_name}"
        )
        return

    if args_text.lower() in ("clear", "reset", "none", "сброс", "очистить"):
        user_state.clear_selected_store(user_id)
        await update.message.reply_text("Selected store cleared. Router will choose automatically.")
        return

    store = gemini_client.find_store_by_name(args_text)
    if not store:
        await update.message.reply_text(f"Store not found: {args_text}")
        return

    user_state.set_selected_store(user_id, store["id"], store.get("name", args_text))
    await update.message.reply_text(f"Selected store: {store.get('name')}")


async def check_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command"""
    if not check_user_allowed(update.effective_user.id):
        await update.message.reply_text("Access denied.")
        return

    selected_store = _get_selected_store_for_user(update.effective_user.id)
    selected_name = selected_store.get("name") if selected_store else "None"

    if drive_client and drive_client.is_configured():
        drive_status = "OK (Service Account)"
    else:
        drive_status = "OK (public URLs only)"

    status_lines = [
        "Status:",
        f"- Gemini API: {'OK' if gemini_client else 'Not configured'}",
        f"- Smart routing: {'OK' if router else 'Not configured'}",
        f"- Query processor: {'OK' if query_processor else 'Not configured'}",
        f"- Google Drive: {drive_status}",
        f"- Stores: {len(gemini_client.stores) if gemini_client else 0}",
        f"- Selected store: {selected_name}",
        f"- Model Flash: {GEMINI_MODEL_FLASH}",
        f"- Model Pro: {GEMINI_MODEL_PRO}",
        "",
        "Smart model selection: Flash for simple, Pro for complex"
    ]

    await update.message.reply_text("\n".join(status_lines))


async def sync_stores(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /sync command - sync stores with API"""
    if not check_user_allowed(update.effective_user.id):
        await update.message.reply_text("Access denied.")
        return

    if not gemini_client:
        await update.message.reply_text("Gemini API not configured.")
        return

    status_msg = await update.message.reply_text("Syncing stores with API...")

    try:
        gemini_client.sync_with_api()
        if router:
            router.reload_library()

        await status_msg.edit_text(
            f"Sync complete!\n"
            f"Total stores: {len(gemini_client.stores)}"
        )
    except Exception as e:
        await status_msg.edit_text(f"Sync error: {str(e)[:200]}")


async def delete_store(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /delete command - delete a store (admin only)"""
    user_id = update.effective_user.id

    if not check_user_allowed(user_id):
        await update.message.reply_text("Access denied.")
        return

    if not is_admin(user_id):
        await update.message.reply_text("Only admin can delete stores.")
        return

    if not gemini_client:
        await update.message.reply_text("Gemini API not configured.")
        return

    args_text = re.sub(r'^/delete\s*', '', update.message.text, flags=re.IGNORECASE).strip()

    if not args_text:
        await update.message.reply_text("Usage: /delete <store_name>")
        return

    store = gemini_client.find_store_by_name(args_text)
    if not store:
        await update.message.reply_text(f"Store not found: {args_text}")
        return

    if gemini_client.delete_store(store["id"]):
        user_state.clear_store_for_all(store["id"])
        if router:
            router.reload_library()
        await update.message.reply_text(f"Deleted: {args_text}")
    else:
        await update.message.reply_text("Failed to delete. Check logs.")


async def rename_store(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /rename command - rename a store (admin only)"""
    user_id = update.effective_user.id

    if not check_user_allowed(user_id):
        await update.message.reply_text("Access denied.")
        return

    if not is_admin(user_id):
        await update.message.reply_text("Only admin can rename stores.")
        return

    if not gemini_client:
        await update.message.reply_text("Gemini API not configured.")
        return

    args_text = re.sub(r'^/rename\\s*', '', update.message.text, flags=re.IGNORECASE).strip()
    if not args_text:
        await update.message.reply_text(
            "Usage: /rename <old_name> | <new_name>\n"
            "Example: /rename ТендерА | ТендерА (обновлено)"
        )
        return

    old_name = None
    new_name = None

    if "|" in args_text:
        parts = [p.strip() for p in args_text.split("|", 1)]
        if len(parts) == 2:
            old_name, new_name = parts
    else:
        # Try common separators
        for sep in ("->", "—", "–", " to ", " в ", " на "):
            if sep in args_text:
                parts = [p.strip() for p in args_text.split(sep, 1)]
                if len(parts) == 2:
                    old_name, new_name = parts
                break

    if not old_name or not new_name:
        await update.message.reply_text(
            "Could not parse names. Use format:\n"
            "/rename <old_name> | <new_name>"
        )
        return

    store = gemini_client.find_store_by_name(old_name)
    if not store:
        await update.message.reply_text(f"Store not found: {old_name}")
        return

    store_name_before = store.get("name", old_name)

    if gemini_client.update_store_metadata(store["id"], name=new_name):
        # Update user state if this store was selected
        selected = user_state.get_selected_store(user_id)
        if selected and selected.get("selected_store_id") == store["id"]:
            user_state.set_selected_store(user_id, store["id"], new_name)

        if router:
            router.reload_library()

        await update.message.reply_text(
            f"Renamed store:\n"
            f"{store_name_before} -> {new_name}"
        )
    else:
        await update.message.reply_text("Failed to rename store. Check logs.")


async def handle_think(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /think command - answer with thinking mode"""
    user_id = update.effective_user.id

    if not check_user_allowed(user_id):
        await update.message.reply_text("Access denied.")
        return

    if not gemini_client:
        await update.message.reply_text("Gemini API not configured.")
        return

    args_text = re.sub(r'^/think\s*', '', update.message.text, flags=re.IGNORECASE).strip()

    if not args_text:
        await update.message.reply_text(
            "Usage: /think <question>\n\n"
            "Uses Gemini thinking mode for complex reasoning."
        )
        return

    if not gemini_client.stores:
        await update.message.reply_text("No knowledge stores available.")
        return

    await update.message.chat.send_action("typing")
    status_msg = await update.message.reply_text("Thinking deeply...")

    try:
        # Route the question
        if router and len(gemini_client.stores) > 1:
            selected, reasoning = router.route_with_reasoning(args_text, max_notebooks=1)
            store = selected[0] if selected else gemini_client.stores[0]
        else:
            store = gemini_client.stores[0]

        await status_msg.edit_text(f"Thinking about: {store.get('name')}...")

        # Get answer with high thinking level
        answer, thinking = gemini_client.ask_with_thinking(
            store["id"],
            args_text,
            thinking_level="high"
        )

        if answer:
            response_text = f"Store: {store.get('name')}\n\n{answer}"

            if len(response_text) > 4000:
                parts = [response_text[i:i+4000] for i in range(0, len(response_text), 4000)]
                await status_msg.edit_text(parts[0])
                for part in parts[1:]:
                    await update.message.reply_text(part)
            else:
                await status_msg.edit_text(response_text)
        else:
            await status_msg.edit_text("No answer received from thinking mode.")

    except Exception as e:
        logger.exception("Error in thinking mode")
        await status_msg.edit_text(f"Error: {str(e)[:500]}")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle photo messages - analyze images with Gemini Vision"""
    user_id = update.effective_user.id

    if not check_user_allowed(user_id):
        await update.message.reply_text("Access denied.")
        return

    if not gemini_client:
        await update.message.reply_text("Gemini API not configured.")
        return

    # Get the largest photo
    photo = update.message.photo[-1]

    # Get prompt from caption or use default
    caption = update.message.caption
    prompt = caption.strip() if caption else IMAGE_DEFAULT_PROMPT

    await update.message.chat.send_action("typing")
    status_msg = await update.message.reply_text("Analyzing image...")

    try:
        # Download photo
        file = await photo.get_file()
        temp_path = _make_temp_path("photo", ".jpg")
        await file.download_to_drive(temp_path)

        # Analyze with Gemini
        result = gemini_client.analyze_image(temp_path, prompt, model=GEMINI_MODEL)

        # Clean up
        temp_path.unlink(missing_ok=True)

        if result:
            if len(result) > 4000:
                parts = [result[i:i+4000] for i in range(0, len(result), 4000)]
                await status_msg.edit_text(parts[0])
                for part in parts[1:]:
                    await update.message.reply_text(part)
            else:
                await status_msg.edit_text(result)
        else:
            await status_msg.edit_text("Could not analyze the image.")

    except Exception as e:
        logger.exception("Error analyzing photo")
        await status_msg.edit_text(f"Error: {str(e)[:500]}")


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle voice messages - transcribe and process as question"""
    user_id = update.effective_user.id

    if not check_user_allowed(user_id):
        await update.message.reply_text("Access denied.")
        return

    if not gemini_client:
        await update.message.reply_text("Gemini API not configured.")
        return

    voice = update.message.voice

    await update.message.chat.send_action("typing")
    status_msg = await update.message.reply_text("Transcribing voice...")

    try:
        # Download voice message
        file = await voice.get_file()
        temp_path = _make_temp_path("voice", ".ogg")
        await file.download_to_drive(temp_path)

        # Transcribe with Gemini
        transcription = gemini_client.transcribe_voice(temp_path, model=GEMINI_MODEL)

        # Clean up
        temp_path.unlink(missing_ok=True)

        if not transcription:
            await status_msg.edit_text("Could not transcribe voice message.")
            return

        await status_msg.edit_text(f"Transcribed: {transcription}\n\nProcessing...")

        # Process transcription as a question if there are stores
        if not gemini_client.stores:
            await status_msg.edit_text(
                f"Transcribed: {transcription}\n\n"
                "No knowledge stores available to answer the question."
            )
            return

        # Process with Pro model for understanding
        processed = query_processor.process_query(
            question=transcription,
            available_stores=gemini_client.stores,
            conversation_context=""
        )

        # Select model based on complexity
        if processed.complexity == "complex":
            voice_model = GEMINI_MODEL_PRO
        else:
            voice_model = GEMINI_MODEL_FLASH

        # Route to best store
        if router and len(gemini_client.stores) > 1:
            selected, reasoning = router.route_with_reasoning(
                processed.optimized_prompt,
                max_notebooks=1
            )
            store = selected[0] if selected else gemini_client.stores[0]
        else:
            store = gemini_client.stores[0]

        answer = gemini_client.ask_question(
            store["id"],
            processed.optimized_prompt,  # Use optimized prompt
            model=voice_model
        )

        if answer:
            response_text = f"Voice: {transcription}\n\n{answer}"
            if len(response_text) > 4000:
                parts = [response_text[i:i+4000] for i in range(0, len(response_text), 4000)]
                await status_msg.edit_text(parts[0])
                for part in parts[1:]:
                    await update.message.reply_text(part)
            else:
                await status_msg.edit_text(response_text)
        else:
            await status_msg.edit_text(
                f"Transcribed: {transcription}\n\n"
                "Could not find an answer in the knowledge stores."
            )

    except Exception as e:
        logger.exception("Error handling voice")
        await status_msg.edit_text(f"Error: {str(e)[:500]}")


async def set_sync(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /setsync command - set URLs for auto-sync (admin only)"""
    user_id = update.effective_user.id

    if not check_user_allowed(user_id):
        await update.message.reply_text("Access denied.")
        return

    if not is_admin(user_id):
        await update.message.reply_text("Only admin can configure sync.")
        return

    if not gemini_client:
        await update.message.reply_text("Gemini API not configured.")
        return

    # Parse: /setsync <store_name> <url1> [url2] ...
    message_text = update.message.text
    args_text = re.sub(r'^/setsync\s*', '', message_text, flags=re.IGNORECASE).strip()

    if not args_text:
        await update.message.reply_text(
            "Usage: /setsync <store_name> <url1> [url2] ...\n\n"
            "Example:\n"
            "/setsync MyStore https://docs.google.com/document/d/xxx\n\n"
            "URLs will be synced daily at 3:00 AM."
        )
        return

    parts = args_text.split(None, 1)
    store_name = parts[0]
    urls_text = parts[1] if len(parts) > 1 else ""

    store = gemini_client.find_store_by_name(store_name)
    if not store:
        await update.message.reply_text(f"Store not found: {store_name}")
        return

    # Extract URLs
    urls = GoogleDriveClient.extract_all_urls(urls_text)
    if not urls:
        await update.message.reply_text(
            "No valid Google URLs found.\n"
            "Supported: docs.google.com, drive.google.com"
        )
        return

    url_list = [url for url, _, _ in urls]

    if gemini_client.set_sync_urls(store["id"], url_list):
        await update.message.reply_text(
            f"Sync configured!\n\n"
            f"Store: {store_name}\n"
            f"URLs: {len(url_list)}\n"
            f"Auto-sync: Enabled (daily at 3:00 AM)\n\n"
            f"Use /syncnow {store_name} to sync immediately."
        )
    else:
        await update.message.reply_text("Failed to configure sync. Check logs.")


async def sync_now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /syncnow command - force sync stores (admin only)"""
    user_id = update.effective_user.id

    if not check_user_allowed(user_id):
        await update.message.reply_text("Access denied.")
        return

    if not is_admin(user_id):
        await update.message.reply_text("Only admin can sync stores.")
        return

    if not gemini_client:
        await update.message.reply_text("Gemini API not configured.")
        return

    # Parse: /syncnow [store_name]
    args_text = re.sub(r'^/syncnow\s*', '', update.message.text, flags=re.IGNORECASE).strip()

    if args_text:
        # Sync specific store
        store = gemini_client.find_store_by_name(args_text)
        if not store:
            await update.message.reply_text(f"Store not found: {args_text}")
            return
        stores_to_sync = [store]
    else:
        # Sync all stores with auto_sync enabled
        stores_to_sync = gemini_client.get_stores_for_sync()
        if not stores_to_sync:
            await update.message.reply_text(
                "No stores configured for sync.\n"
                "Use /setsync to configure."
            )
            return

    status_msg = await update.message.reply_text(
        f"Syncing {len(stores_to_sync)} store(s)..."
    )

    results = []

    for store in stores_to_sync:
        sync_urls = store.get("sync_urls", [])
        if not sync_urls:
            results.append(f"- {store.get('name')}: No sync URLs configured")
            continue

        await status_msg.edit_text(f"Syncing {store.get('name')}...")

        success_count = 0
        error_count = 0

        import tempfile
        temp_dir = Path(tempfile.mkdtemp(prefix="sync_"))

        try:
            for url in sync_urls:
                extracted = GoogleDriveClient.extract_file_id(url)
                if not extracted:
                    error_count += 1
                    continue

                file_id, file_type = extracted

                if file_type == 'folder':
                    if drive_client and drive_client.is_configured():
                        downloaded = drive_client.download_folder(file_id, temp_dir)
                        for file_path, file_name in downloaded:
                            if gemini_client.upload_file(store["id"], file_path, file_name):
                                success_count += 1
                            else:
                                error_count += 1
                            file_path.unlink(missing_ok=True)
                    else:
                        error_count += 1
                else:
                    file_path = drive_client.download_file(file_id, temp_dir, file_type=file_type)
                    if file_path:
                        if gemini_client.upload_file(store["id"], file_path, file_path.name):
                            success_count += 1
                        else:
                            error_count += 1
                        file_path.unlink(missing_ok=True)
                    else:
                        error_count += 1

            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

            gemini_client.update_last_sync(store["id"])
            results.append(f"- {store.get('name')}: +{success_count} files, {error_count} errors")

        except Exception as e:
            logger.error(f"Sync error for {store.get('name')}: {e}")
            results.append(f"- {store.get('name')}: Error - {str(e)[:50]}")
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    await status_msg.edit_text(
        f"Sync complete!\n\n" + "\n".join(results)
    )


async def auto_sync_callback(context: ContextTypes.DEFAULT_TYPE):
    """JobQueue callback for daily auto-sync"""
    logger.info("Running scheduled auto-sync...")

    stores_to_sync = gemini_client.get_stores_for_sync()
    if not stores_to_sync:
        logger.info("No stores configured for auto-sync")
        return

    for store in stores_to_sync:
        sync_urls = store.get("sync_urls", [])
        if not sync_urls:
            continue

        logger.info(f"Auto-syncing {store.get('name')} ({len(sync_urls)} URLs)...")

        import tempfile
        temp_dir = Path(tempfile.mkdtemp(prefix="autosync_"))
        success_count = 0
        error_count = 0

        try:
            for url in sync_urls:
                extracted = GoogleDriveClient.extract_file_id(url)
                if not extracted:
                    error_count += 1
                    continue

                file_id, file_type = extracted

                if file_type == 'folder':
                    if drive_client and drive_client.is_configured():
                        downloaded = drive_client.download_folder(file_id, temp_dir)
                        for file_path, file_name in downloaded:
                            if gemini_client.upload_file(store["id"], file_path, file_name):
                                success_count += 1
                            else:
                                error_count += 1
                            file_path.unlink(missing_ok=True)
                    else:
                        error_count += 1
                else:
                    file_path = drive_client.download_file(file_id, temp_dir, file_type=file_type)
                    if file_path:
                        if gemini_client.upload_file(store["id"], file_path, file_path.name):
                            success_count += 1
                        else:
                            error_count += 1
                        file_path.unlink(missing_ok=True)
                    else:
                        error_count += 1

            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

            gemini_client.update_last_sync(store["id"])
            logger.info(f"Auto-sync {store.get('name')}: +{success_count} files, {error_count} errors")

        except Exception as e:
            logger.error(f"Auto-sync error for {store.get('name')}: {e}")
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


async def compare_stores(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /compare command - compare two stores by topic"""
    user_id = update.effective_user.id

    if not check_user_allowed(user_id):
        await update.message.reply_text("Access denied.")
        return

    if not gemini_client:
        await update.message.reply_text("Gemini API not configured.")
        return

    # Parse: /compare <store1> <store2> <topic>
    message_text = update.message.text
    args_text = re.sub(r'^/compare\s*', '', message_text, flags=re.IGNORECASE).strip()

    if not args_text:
        stores_list = ", ".join([s.get("name", "?") for s in gemini_client.stores[:5]])
        await update.message.reply_text(
            "Usage: /compare <store1> <store2> <topic>\n\n"
            "Example:\n"
            "/compare ТендерА ТендерБ водопонижение\n\n"
            f"Available stores: {stores_list}"
        )
        return

    parts = args_text.split(None, 2)
    if len(parts) < 3:
        await update.message.reply_text(
            "Please provide two store names and a topic.\n"
            "Example: /compare Store1 Store2 земляные работы"
        )
        return

    store_name_1, store_name_2, topic = parts

    store_1 = gemini_client.find_store_by_name(store_name_1)
    store_2 = gemini_client.find_store_by_name(store_name_2)

    if not store_1:
        await update.message.reply_text(f"Store not found: {store_name_1}")
        return

    if not store_2:
        await update.message.reply_text(f"Store not found: {store_name_2}")
        return

    await update.message.chat.send_action("typing")
    status_msg = await update.message.reply_text(
        f"Comparing {store_1.get('name')} vs {store_2.get('name')}\n"
        f"Topic: {topic}\n\n"
        "This may take a moment..."
    )

    try:
        # Comparisons always use Pro model (complex analysis)
        result = gemini_client.compare_stores(
            store_1["id"],
            store_2["id"],
            topic,
            model=GEMINI_MODEL_PRO
        )

        if result:
            header = f"Сравнение: {store_1.get('name')} vs {store_2.get('name')}\n"
            header += f"Тема: {topic}\n\n"
            full_response = header + result

            if len(full_response) > 4000:
                parts = [full_response[i:i+4000] for i in range(0, len(full_response), 4000)]
                await status_msg.edit_text(parts[0])
                for part in parts[1:]:
                    await update.message.reply_text(part)
            else:
                await status_msg.edit_text(full_response)
        else:
            await status_msg.edit_text(
                f"Could not generate comparison for topic: {topic}"
            )

    except Exception as e:
        logger.exception("Error in compare")
        await status_msg.edit_text(f"Error: {str(e)[:500]}")


async def export_response(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /export command - export last response to PDF/DOCX"""
    user_id = update.effective_user.id

    if not check_user_allowed(user_id):
        await update.message.reply_text("Access denied.")
        return

    # Parse: /export [question] or just /export
    args_text = re.sub(r'^/export\s*', '', update.message.text, flags=re.IGNORECASE).strip()

    last_response = context.user_data.get("last_response")

    if args_text:
        # User provided a new question - process it and export
        await update.message.reply_text(
            "Processing your question for export..."
        )
        # Reuse normal flow: answer the question, then user can export
        context.user_data["force_question_mode"] = True
        update.message.text = args_text
        await handle_question(update, context)
        return

    if not last_response:
        await update.message.reply_text(
            "No previous response to export.\n"
            "Ask a question first, then use /export."
        )
        return

    # Show export options
    keyboard = [
        [
            InlineKeyboardButton("PDF", callback_data="export_pdf"),
            InlineKeyboardButton("DOCX", callback_data="export_docx"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        f"Export response about:\n"
        f"Q: {last_response.get('question', '')[:100]}...\n\n"
        f"Choose format:",
        reply_markup=reply_markup
    )


async def _send_export_file(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    export_format: str,
    question: str,
    answer: str,
    store_name: str
) -> bool:
    """Generate and send export file. Returns True on success."""
    title = question[:50] if question else "Export"

    if export_format == "pdf":
        file_path = export_client.export_to_pdf(
            content=answer,
            title=title,
            question=question,
            store_name=store_name
        )
    else:
        file_path = export_client.export_to_docx(
            content=answer,
            title=title,
            question=question,
            store_name=store_name
        )

    if file_path and file_path.exists():
        with open(file_path, "rb") as f:
            await context.bot.send_document(
                chat_id=chat_id,
                document=f,
                filename=file_path.name,
                caption=f"Export: {title}"
            )
        file_path.unlink(missing_ok=True)
        return True

    return False


async def handle_export_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle export format selection callback"""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    if not check_user_allowed(user_id):
        await query.edit_message_text("Access denied.")
        return

    last_response = context.user_data.get("last_response")
    if not last_response:
        await query.edit_message_text("Response expired. Ask a new question.")
        return

    export_format = query.data.replace("export_", "")

    await query.edit_message_text(f"Generating {export_format.upper()}...")

    question = last_response.get("question", "")
    answer = last_response.get("answer", "")
    store_name = last_response.get("store", "")

    ok = await _send_export_file(
        context=context,
        chat_id=query.message.chat_id,
        export_format=export_format,
        question=question,
        answer=answer,
        store_name=store_name
    )

    if ok:
        await query.delete_message()
    else:
        await query.edit_message_text(
            f"Failed to generate {export_format.upper()}.\n"
            "Make sure reportlab (PDF) or python-docx (DOCX) is installed."
        )


def get_export_keyboard():
    """Get inline keyboard for export options."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("PDF", callback_data="export_pdf"),
            InlineKeyboardButton("DOCX", callback_data="export_docx"),
        ]
    ])


async def clear_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /clear command - clear conversation history"""
    user_id = update.effective_user.id

    if not check_user_allowed(user_id):
        await update.message.reply_text("Access denied.")
        return

    memory_client.clear_history(user_id)

    await update.message.reply_text(
        "История диалога очищена.\n"
        "Бот забыл контекст предыдущих вопросов."
    )


async def handle_folder_link(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle Google Drive folder link - auto-create store with AI-generated name"""
    user_id = update.effective_user.id
    message_text = update.message.text.strip()

    # Extract folder URL
    urls = GoogleDriveClient.extract_all_urls(message_text)
    folder_urls = [(url, fid, ftype) for url, fid, ftype in urls if ftype == 'folder']

    if not folder_urls:
        # Log for debugging
        logger.warning(f"No folder URL detected in: {message_text[:100]}...")
        logger.warning(f"All extracted URLs: {urls}")
        return False  # Not a folder link

    if not is_admin(user_id):
        await update.message.reply_text(
            "Only admin can create stores from folder links."
        )
        return True

    if not gemini_client:
        await update.message.reply_text("Gemini API not configured.")
        return True

    url, folder_id, _ = folder_urls[0]

    logger.info(f"Processing folder link: {url}, folder_id: {folder_id}")

    await update.message.chat.send_action("typing")
    status_msg = await update.message.reply_text(
        f"Detected Google Drive folder.\n"
        f"Folder ID: {folder_id[:20]}...\n"
        "Creating new tender store..."
    )

    # Create temporary store
    import uuid
    temp_name = f"Tender_{uuid.uuid4().hex[:8]}"

    await status_msg.edit_text(f"Creating store '{temp_name}'...")

    result = gemini_client.create_store(temp_name, "Analyzing...")
    if not result:
        await status_msg.edit_text("Failed to create store. Check logs.")
        return True

    store_id = result["id"]

    # Download and upload files from folder
    if not drive_client or not drive_client.is_configured():
        await status_msg.edit_text(
            "Google Drive Service Account required for folder access.\n"
            "Configure service_account.json"
        )
        return True

    await status_msg.edit_text("Downloading files from folder...")

    import tempfile
    temp_dir = Path(tempfile.mkdtemp(prefix="folder_"))

    try:
        downloaded = drive_client.download_folder(folder_id, temp_dir)

        if not downloaded:
            await status_msg.edit_text("No files found in folder or access denied.")
            gemini_client.delete_store(store_id)
            return True

        await status_msg.edit_text(f"Uploading {len(downloaded)} files...")

        success_count = 0
        for file_path, file_name in downloaded:
            if gemini_client.upload_file(store_id, file_path, file_name, source_url=url):
                success_count += 1
            file_path.unlink(missing_ok=True)

        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

        if success_count == 0:
            await status_msg.edit_text("Failed to upload files.")
            gemini_client.delete_store(store_id)
            return True

        # Analyze content with Gemini Pro to get name and description
        await status_msg.edit_text(
            f"Uploaded {success_count} files.\n"
            "Analyzing content to determine tender name..."
        )

        analysis = gemini_client.analyze_store_content(store_id, model=GEMINI_MODEL_PRO)

        if analysis:
            tender_name = analysis.get("name", temp_name)
            tender_desc = analysis.get("description", "")

            gemini_client.update_store_metadata(store_id, tender_name, tender_desc)

            if router:
                router.reload_library()

            await status_msg.edit_text(
                f"Store created!\n\n"
                f"Name: {tender_name}\n"
                f"Description: {tender_desc}\n"
                f"Documents: {success_count}\n\n"
                f"You can now ask questions about this tender."
            )
        else:
            if router:
                router.reload_library()

            await status_msg.edit_text(
                f"Store created!\n\n"
                f"Name: {temp_name}\n"
                f"Documents: {success_count}\n\n"
                f"Could not auto-detect tender name.\n"
                f"You can rename with /rename command."
            )

    except Exception as e:
        logger.exception("Error processing folder")
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        await status_msg.edit_text(f"Error: {str(e)[:300]}")

    return True


def _pick_action(processed, question: str) -> Tuple[Optional[str], dict]:
    action = None
    action_args = {}

    if processed and processed.action and processed.action != "none":
        if processed.confidence >= ACTION_CONFIDENCE_THRESHOLD:
            action = processed.action
            action_args = processed.action_args or {}

    if not action:
        action, action_args = infer_action_from_text(question)

    if not isinstance(action_args, dict):
        action_args = {}

    return action, action_args


def _build_command_text(command: str, *parts: str) -> str:
    clean_parts = [p for p in parts if p]
    if clean_parts:
        return f"/{command} " + " ".join(clean_parts)
    return f"/{command}"


async def _dispatch_action_intent(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    action: str,
    action_args: dict,
    question_text: str
) -> bool:
    """Execute command-like action intents. Returns True if handled."""
    if not action:
        return False

    # Simple actions
    if action in ("list_stores", "stores"):
        await list_stores(update, context)
        return True
    if action == "status":
        await check_status(update, context)
        return True
    if action == "clear_memory":
        await clear_memory(update, context)
        return True
    if action == "help":
        await start(update, context)
        return True

    # Select store
    if action == "select_store":
        store_name = action_args.get("store_name") or extract_target_store_hint(question_text)
        if not store_name:
            await update.message.reply_text(
                "Please specify a store name.\nExample: \"Выбери тендер Дубровка\""
            )
            return True
        update.message.text = _build_command_text("select", store_name)
        await select_store(update, context)
        return True

    # Add store (admin)
    if action == "add_store":
        store_name = action_args.get("store_name") or action_args.get("name")
        description = action_args.get("description", "")
        if not store_name:
            await update.message.reply_text(
                "Please specify a store name.\nExample: \"Создай тендер Дубровка | Земляные работы\""
            )
            return True
        update.message.text = f"/add {store_name} | {description}" if description else f"/add {store_name}"
        await add_store(update, context)
        return True

    # Delete store (admin)
    if action == "delete_store":
        store_name = action_args.get("store_name") or extract_target_store_hint(question_text)
        if not store_name:
            await update.message.reply_text(
                "Please specify a store name to delete.\nExample: \"Удалить тендер Дубровка\""
            )
            return True
        update.message.text = _build_command_text("delete", store_name)
        await delete_store(update, context)
        return True

    # Rename store (admin)
    if action == "rename_store":
        old_name = action_args.get("old_name")
        new_name = action_args.get("new_name")
        if not old_name or not new_name:
            await update.message.reply_text(
                "Please provide old and new store names.\n"
                "Example: \"Переименуй тендер Дубровка в Дубровка (2026)\""
            )
            return True
        update.message.text = f"/rename {old_name} | {new_name}"
        await rename_store(update, context)
        return True

    # Export
    if action == "export":
        export_format = action_args.get("format")
        if export_format in ("pdf", "docx"):
            last_response = context.user_data.get("last_response")
            if not last_response:
                await update.message.reply_text(
                    "No previous response to export.\nAsk a question first."
                )
                return True

            ok = await _send_export_file(
                context=context,
                chat_id=update.effective_chat.id,
                export_format=export_format,
                question=last_response.get("question", ""),
                answer=last_response.get("answer", ""),
                store_name=last_response.get("store", "")
            )

            if not ok:
                await update.message.reply_text(
                    f"Failed to generate {export_format.upper()} export."
                )
        else:
            await export_response(update, context)
        return True

    # Sync related (admin)
    if action == "sync_now":
        store_name = action_args.get("store_name")
        update.message.text = _build_command_text("syncnow", store_name) if store_name else "/syncnow"
        await sync_now(update, context)
        return True

    if action == "set_sync":
        store_name = action_args.get("store_name") or extract_target_store_hint(question_text)
        urls = action_args.get("urls")
        if isinstance(urls, str):
            urls = [urls]
        urls = urls or [u for u, _, _ in GoogleDriveClient.extract_all_urls(question_text)]
        if not store_name or not urls:
            await update.message.reply_text(
                "Please specify store and URLs.\n"
                "Example: \"Настрой синхронизацию для Дубровка https://docs.google.com/...\""
            )
            return True
        update.message.text = f"/setsync {store_name} " + " ".join(urls)
        await set_sync(update, context)
        return True

    if action == "upload_url":
        store_name = action_args.get("store_name") or extract_target_store_hint(question_text)
        urls = action_args.get("urls")
        if isinstance(urls, str):
            urls = [urls]
        urls = urls or [u for u, _, _ in GoogleDriveClient.extract_all_urls(question_text)]
        if not store_name or not urls:
            await update.message.reply_text(
                "Please specify store and Google URL(s).\n"
                "Example: \"Загрузи в Дубровка https://docs.google.com/...\""
            )
            return True
        update.message.text = f"/uploadurl {store_name} " + " ".join(urls)
        await upload_from_url(update, context)
        return True

    if action == "upload_file":
        store_name = action_args.get("store_name") or extract_target_store_hint(question_text)
        if not store_name:
            await update.message.reply_text(
                "Please specify store name for upload.\n"
                "Example: \"Загрузи файл в тендер Дубровка\""
            )
            return True
        update.message.text = _build_command_text("upload", store_name)
        await upload_file(update, context)
        return True

    return False


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user questions with AI-powered understanding and ultrathinking"""
    user_id = update.effective_user.id

    if not check_user_allowed(user_id):
        await update.message.reply_text("Access denied.")
        return

    question = update.message.text.strip()
    if not question:
        return

    # Check if this is a Google Drive folder link (admin only)
    # Use proper URL extraction instead of simple string check
    detected_urls = GoogleDriveClient.extract_all_urls(question)
    has_folder_link = any(ftype == 'folder' for _, _, ftype in detected_urls)

    if has_folder_link:
        handled = await handle_folder_link(update, context)
        if handled:
            return

    if not gemini_client:
        await update.message.reply_text("Gemini API not configured.")
        return

    if not gemini_client.stores:
        await update.message.reply_text(
            "No knowledge stores available.\n"
            "Admin can create with /add command."
        )
        return

    await update.message.chat.send_action("typing")

    status_msg = await update.message.reply_text("Analyzing your question...")

    try:
        # Get conversation context for better understanding
        # Use "global" context for cross-store queries
        conversation_context = memory_client.get_context_prompt(user_id, "global")

        # Process query with ultrathinking to understand intent
        await status_msg.edit_text("Understanding your request...")

        processed = query_processor.process_query(
            question=question,
            available_stores=gemini_client.stores,
            conversation_context=conversation_context
        )

        logger.info(f"Query type: {processed.query_type}, complexity: {processed.complexity}, "
                   f"intent: {processed.user_intent}, confidence: {processed.confidence}")

        # Handle command-like action intents (natural language)
        force_question_mode = context.user_data.pop("force_question_mode", False)
        if not force_question_mode:
            action, action_args = _pick_action(processed, question)

            # Special case: export with an explicit question
            if action == "export" and action_args.get("question"):
                export_question = str(action_args.get("question")).strip()
                if not export_question:
                    await status_msg.edit_text("Please provide a question for export.")
                    return

                # Re-process the actual question for proper routing
                question = export_question
                processed = query_processor.process_query(
                    question=question,
                    available_stores=gemini_client.stores,
                    conversation_context=conversation_context
                )

                context.user_data["export_after_answer"] = action_args.get("format")
            else:
                handled = await _dispatch_action_intent(update, context, action, action_args, question)
                if handled:
                    return

        # Select model based on complexity
        # Simple/Medium -> Flash (fast, cheap)
        # Complex -> Pro (smart, thorough)
        if processed.complexity == "complex":
            query_model = GEMINI_MODEL_PRO
            logger.info(f"Using Pro model for complex query")
        else:
            query_model = GEMINI_MODEL_FLASH
            logger.info(f"Using Flash model for {processed.complexity} query")

        # Show what AI understood
        intent_text = f"Понял: {processed.user_intent}" if processed.user_intent else ""

        # Handle different query types
        if processed.query_type == "web_search":
            await status_msg.edit_text(f"{intent_text}\n\nSearching the web...")

            answer = gemini_client.ask_with_web_search(
                processed.optimized_prompt,
                model=query_model
            )

            if answer:
                answer = f"[Веб-поиск]\n\n{answer}"
                memory_client.add_message(user_id, "web", "user", question)
                memory_client.add_message(user_id, "web", "assistant", answer)

                await _send_answer(status_msg, update, answer, context, question, "web")
            else:
                await status_msg.edit_text("Не удалось выполнить веб-поиск.")
            return

        if processed.query_type == "multistore":
            await status_msg.edit_text(
                f"{intent_text}\n\nSearching across {len(gemini_client.stores)} stores..."
            )

            store_ids = [s["id"] for s in gemini_client.stores]
            results = gemini_client.ask_multistore_parallel(
                store_ids,
                processed.optimized_prompt,  # Use optimized prompt
                model=query_model
            )

            answer = gemini_client.format_multistore_response(results)
            memory_client.add_message(user_id, "global", "user", question)
            memory_client.add_message(user_id, "global", "assistant", answer)

            await _send_answer(status_msg, update, answer, context, question, "multistore")
            return

        if processed.query_type == "compare":
            # Handle comparison
            if processed.target_stores and len(processed.target_stores) >= 2:
                store_1 = gemini_client.find_store_by_name(processed.target_stores[0])
                store_2 = gemini_client.find_store_by_name(processed.target_stores[1])

                if store_1 and store_2:
                    await status_msg.edit_text(
                        f"{intent_text}\n\n"
                        f"Comparing {store_1.get('name')} vs {store_2.get('name')}..."
                    )

                    # Comparisons always use Pro model (complex task)
                    result = gemini_client.compare_stores(
                        store_1["id"],
                        store_2["id"],
                        processed.compare_topic or processed.optimized_prompt,
                        model=GEMINI_MODEL_PRO
                    )

                    if result:
                        answer = f"Сравнение: {store_1.get('name')} vs {store_2.get('name')}\n\n{result}"
                        await _send_answer(status_msg, update, answer, context, question, "compare")
                        return

            # Fallback if comparison failed
            await status_msg.edit_text("Не удалось определить stores для сравнения.")
            return

        # Single store query (default)
        # Prefer explicit target store from AI or user selection
        target_store_name = processed.target_store or extract_target_store_hint(question)
        store = None

        if target_store_name:
            store = gemini_client.find_store_by_name(target_store_name)

        if not store:
            selected_store = _get_selected_store_for_user(user_id)
            if selected_store:
                store = selected_store

        # Route to best store if multiple available and no explicit selection
        if not store:
            if router and len(gemini_client.stores) > 1:
                selected, reasoning = router.route_with_reasoning(
                    processed.optimized_prompt,
                    max_notebooks=1
                )
                if selected:
                    store = selected[0]
                else:
                    store = gemini_client.stores[0]
            else:
                store = gemini_client.stores[0]

        await status_msg.edit_text(
            f"{intent_text}\n\n"
            f"Querying {store.get('name')}..."
        )

        # Get store-specific conversation context
        store_context = memory_client.get_context_prompt(user_id, store["id"])

        # Build final prompt with context
        if store_context:
            final_prompt = f"{store_context}\n{processed.optimized_prompt}"
        else:
            final_prompt = processed.optimized_prompt

        # Save user message to memory
        memory_client.add_message(user_id, store["id"], "user", question)

        # Get answer with or without sources
        if processed.include_sources:
            answer = gemini_client.ask_with_sources(
                store["id"],
                final_prompt,
                model=query_model
            )
        else:
            answer = gemini_client.ask_question(
                store["id"],
                final_prompt,
                model=query_model
            )

        if answer:
            memory_client.add_message(user_id, store["id"], "assistant", answer)
            await _send_answer(status_msg, update, answer, context, question, store.get("name", ""))
        else:
            await status_msg.edit_text(
                "No answer received.\n"
                "The store might be empty or the question couldn't be answered."
            )

    except Exception as e:
        logger.exception("Error handling question")
        await status_msg.edit_text(f"Error: {str(e)[:500]}")


async def _send_answer(status_msg, update, answer, context, question, store_name):
    """Helper to send answer with export buttons and handle long messages."""
    # Save for export
    context.user_data["last_response"] = {
        "question": question,
        "answer": answer,
        "store": store_name,
        "timestamp": datetime.now().isoformat()
    }

    if len(answer) > 4000:
        parts = [answer[i:i+4000] for i in range(0, len(answer), 4000)]
        await status_msg.edit_text(parts[0])
        for part in parts[1:]:
            await update.message.reply_text(part)
        await update.message.reply_text("Export:", reply_markup=get_export_keyboard())
    else:
        await status_msg.edit_text(answer, reply_markup=get_export_keyboard())

    # Auto-export if requested in the same flow
    export_after = context.user_data.pop("export_after_answer", None)
    if export_after in ("pdf", "docx"):
        ok = await _send_export_file(
            context=context,
            chat_id=update.effective_chat.id,
            export_format=export_after,
            question=question,
            answer=answer,
            store_name=store_name
        )
        if not ok:
            await update.message.reply_text(
                f"Failed to generate {export_after.upper()} export."
            )


async def send_notification(context: ContextTypes.DEFAULT_TYPE, message: str, parse_mode: str = None):
    """
    Send notification to the team channel if configured.

    Args:
        context: Telegram context
        message: Message text to send
        parse_mode: Optional parse mode (HTML, Markdown)
    """
    if not NOTIFICATION_CHANNEL_ID:
        return

    try:
        await context.bot.send_message(
            chat_id=NOTIFICATION_CHANNEL_ID,
            text=message,
            parse_mode=parse_mode
        )
        logger.info(f"Notification sent to channel {NOTIFICATION_CHANNEL_ID}")
    except Exception as e:
        logger.error(f"Failed to send notification: {e}")


async def handle_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /summary command - generate executive summary of a tender"""
    user_id = update.effective_user.id

    if not check_user_allowed(user_id):
        await update.message.reply_text("Access denied.")
        return

    if not gemini_client:
        await update.message.reply_text("Gemini API not configured.")
        return

    if not gemini_client.stores:
        await update.message.reply_text(
            "No knowledge stores available.\n"
            "Admin can create with /add command."
        )
        return

    # Get store from args or selected store
    args_text = re.sub(r'^/summary\s*', '', update.message.text, flags=re.IGNORECASE).strip()

    store = None
    if args_text:
        store = gemini_client.find_store_by_name(args_text)
        if not store:
            await update.message.reply_text(f"Store not found: {args_text}")
            return
    else:
        store = _get_selected_store_for_user(user_id)
        if not store:
            await update.message.reply_text(
                "Usage: /summary [store_name]\n\n"
                "Or select a store first with /select command."
            )
            return

    await update.message.chat.send_action("typing")
    status_msg = await update.message.reply_text(
        f"Генерирую саммари для {store.get('name')}...\n"
        "Это может занять некоторое время."
    )

    try:
        # Load summary prompt from library
        summary_prompt = """Роль: эксперт тендерного отдела генподрядчика (Москва). Проанализируй ВСЮ загруженную документацию по тендеру и составь Executive Summary на 1 страницу.

СТРУКТУРА САММАРИ:

1. **ОБЪЕКТ**
   - Название проекта
   - Тип объекта (жилой, коммерческий, промышленный)
   - Адрес/местоположение
   - Площадь/этажность/основные параметры

2. **ЗАКАЗЧИК**
   - Наименование организации
   - Контактные данные (если есть)
   - Тип заказчика (застройщик, генподрядчик, госзаказ)

3. **ОБЪЁМЫ РАБОТ**
   - Основные виды работ
   - Ключевые объёмы (м², м³, т)
   - Особые требования к материалам

4. **СРОКИ**
   - Дата подачи КП
   - Плановое начало работ
   - Срок завершения
   - Ключевые этапы

5. **СТОИМОСТЬ** (если указана)
   - НМЦ или ориентировочная стоимость
   - Условия оплаты
   - Обеспечение контракта

6. **КЛЮЧЕВЫЕ РИСКИ** (топ-5)
   - Технические риски
   - Коммерческие риски
   - Сроковые риски

7. **РЕКОМЕНДАЦИЯ**
   - Участвовать / Не участвовать / Требуется уточнение
   - Краткое обоснование

Для каждого пункта указывай источник (документ, раздел/страница). Если информация отсутствует — укажи 'Не указано в документации'."""

        # Use Pro model for complex analysis
        answer = gemini_client.ask_question(
            store["id"],
            summary_prompt,
            model=GEMINI_MODEL_PRO
        )

        if answer:
            # Save for export
            context.user_data["last_response"] = {
                "question": "Executive Summary тендера",
                "answer": answer,
                "store": store.get("name", ""),
                "timestamp": datetime.now().isoformat()
            }

            # Send notification to channel
            if NOTIFICATION_CHANNEL_ID:
                user_name = update.effective_user.full_name or update.effective_user.username
                await send_notification(
                    context,
                    f"📋 Сгенерировано саммари тендера\n"
                    f"Тендер: {store.get('name')}\n"
                    f"Пользователь: {user_name}",
                )

            if len(answer) > 4000:
                parts = [answer[i:i+4000] for i in range(0, len(answer), 4000)]
                await status_msg.edit_text(parts[0])
                for part in parts[1:]:
                    await update.message.reply_text(part)
                await update.message.reply_text("Export:", reply_markup=get_export_keyboard())
            else:
                await status_msg.edit_text(answer, reply_markup=get_export_keyboard())
        else:
            await status_msg.edit_text(
                "Не удалось сгенерировать саммари.\n"
                "Возможно, в store недостаточно документов."
            )

    except Exception as e:
        logger.exception("Error in summary")
        await status_msg.edit_text(f"Error: {str(e)[:500]}")


async def handle_generate_rfi(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /generate_rfi command - generate RFI letter to customer"""
    user_id = update.effective_user.id

    if not check_user_allowed(user_id):
        await update.message.reply_text("Access denied.")
        return

    if not gemini_client:
        await update.message.reply_text("Gemini API not configured.")
        return

    if not gemini_client.stores:
        await update.message.reply_text(
            "No knowledge stores available.\n"
            "Admin can create with /add command."
        )
        return

    # Parse: /generate_rfi [store_name] [topic]
    args_text = re.sub(r'^/generate_rfi\s*', '', update.message.text, flags=re.IGNORECASE).strip()

    store = None
    topic = ""

    if args_text:
        # Try to parse store name and topic
        parts = args_text.split(None, 1)
        store_candidate = parts[0]
        store = gemini_client.find_store_by_name(store_candidate)

        if store and len(parts) > 1:
            topic = parts[1]
        elif not store:
            # Maybe the whole text is a topic, use selected store
            store = _get_selected_store_for_user(user_id)
            topic = args_text
    else:
        store = _get_selected_store_for_user(user_id)

    if not store:
        await update.message.reply_text(
            "Usage: /generate_rfi [store_name] [topic]\n\n"
            "Examples:\n"
            "/generate_rfi МайПриорити\n"
            "/generate_rfi МайПриорити водопонижение\n"
            "/generate_rfi  (uses selected store)\n\n"
            "Or select a store first with /select command."
        )
        return

    await update.message.chat.send_action("typing")
    status_msg = await update.message.reply_text(
        f"Генерирую RFI письмо для {store.get('name')}...\n"
        f"{'Тема: ' + topic if topic else 'Анализирую всю документацию...'}"
    )

    try:
        # Build RFI prompt
        rfi_prompt = f"""Роль: инженер ПТО генподрядчика, отвечающий за подготовку тендерной документации.

ЗАДАЧА: Сформируй официальное письмо-запрос (RFI — Request for Information) заказчику.
{f'ФОКУС: Сосредоточься на теме: {topic}' if topic else ''}

АНАЛИЗ: Изучи загруженную документацию и выяви:
1. Противоречия между документами
2. Отсутствующую информацию
3. Неоднозначные требования
4. Вопросы по объёмам и срокам

ФОРМАТ ПИСЬМА:

---
Исх. №______ от «__» _______ 2025 г.

Генеральному директору
[Наименование организации-заказчика]
[ФИО]

О запросе разъяснений по тендерной документации
Объект: [Название объекта]

Уважаемый [Имя Отчество]!

В рамках подготовки коммерческого предложения по объекту [название] просим предоставить разъяснения по следующим вопросам:

№ | Раздел документации | Вопрос | Ссылка на документ
1 | [раздел] | [вопрос] | [документ, п.X / стр.Y]
2 | ...

Просим направить ответ в срок до [дата] для своевременной подготовки предложения.

С уважением,
[Должность]
[ФИО]
[Контакты]
---

ВАЖНО:
- Вопросы отсортируй по приоритету: Критичные → Важные → Уточняющие
- Для каждого вопроса укажи конкретную ссылку на документ
- Формулируй вопросы чётко и однозначно
- Количество вопросов: 5-15 (оптимально)"""

        # Use Pro model for complex document analysis
        answer = gemini_client.ask_question(
            store["id"],
            rfi_prompt,
            model=GEMINI_MODEL_PRO
        )

        if answer:
            # Save for export
            context.user_data["last_response"] = {
                "question": f"RFI письмо{' по теме: ' + topic if topic else ''}",
                "answer": answer,
                "store": store.get("name", ""),
                "timestamp": datetime.now().isoformat()
            }

            # Send notification to channel
            if NOTIFICATION_CHANNEL_ID:
                user_name = update.effective_user.full_name or update.effective_user.username
                await send_notification(
                    context,
                    f"📝 Сгенерировано RFI письмо\n"
                    f"Тендер: {store.get('name')}\n"
                    f"{'Тема: ' + topic if topic else ''}\n"
                    f"Пользователь: {user_name}",
                )

            if len(answer) > 4000:
                parts = [answer[i:i+4000] for i in range(0, len(answer), 4000)]
                await status_msg.edit_text(parts[0])
                for part in parts[1:]:
                    await update.message.reply_text(part)
                await update.message.reply_text("Export:", reply_markup=get_export_keyboard())
            else:
                await status_msg.edit_text(answer, reply_markup=get_export_keyboard())
        else:
            await status_msg.edit_text(
                "Не удалось сгенерировать RFI письмо.\n"
                "Возможно, в store недостаточно документов."
            )

    except Exception as e:
        logger.exception("Error in generate_rfi")
        await status_msg.edit_text(f"Error: {str(e)[:500]}")


async def handle_norm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /norm command - search for regulatory document mentions"""
    user_id = update.effective_user.id

    if not check_user_allowed(user_id):
        await update.message.reply_text("Access denied.")
        return

    if not gemini_client:
        await update.message.reply_text("Gemini API not configured.")
        return

    if not gemini_client.stores:
        await update.message.reply_text(
            "No knowledge stores available.\n"
            "Admin can create with /add command."
        )
        return

    # Parse: /norm <norm_code> [store_name]
    args_text = re.sub(r'^/norm\s*', '', update.message.text, flags=re.IGNORECASE).strip()

    if not args_text:
        await update.message.reply_text(
            "Usage: /norm <код_норматива> [store_name]\n\n"
            "Examples:\n"
            "/norm СП 48.13330\n"
            "/norm ГОСТ 27751-2014\n"
            "/norm СНиП 3.02.01-87 МайПриорити\n\n"
            "Without store name, searches in selected store or all stores."
        )
        return

    # Try to extract norm code and optional store name
    parts = args_text.rsplit(None, 1)
    norm_code = args_text
    store = None

    if len(parts) > 1:
        # Check if last part is a store name
        potential_store = gemini_client.find_store_by_name(parts[1])
        if potential_store:
            store = potential_store
            norm_code = parts[0]

    if not store:
        store = _get_selected_store_for_user(user_id)

    await update.message.chat.send_action("typing")

    # If no store selected, search in all stores
    if not store:
        status_msg = await update.message.reply_text(
            f"Поиск норматива '{norm_code}' во всех тендерах..."
        )

        # Search across all stores
        results = []
        for s in gemini_client.stores:
            norm_prompt = f"""Найди ВСЕ упоминания норматива '{norm_code}' в документации.

Для каждого упоминания укажи:
1. Раздел/страница
2. Контекст (2-3 предложения вокруг упоминания)
3. Что требуется по этому нормативу

Если норматив НЕ найден, ответь: 'Норматив {norm_code} не найден в документации.'"""

            answer = gemini_client.ask_question(
                s["id"],
                norm_prompt,
                model=GEMINI_MODEL_FLASH
            )

            if answer and "не найден" not in answer.lower():
                results.append({
                    "store": s.get("name"),
                    "answer": answer
                })

        if results:
            response_text = f"## Норматив: {norm_code}\n\n"
            for r in results:
                response_text += f"### 📁 {r['store']}\n{r['answer']}\n\n"

            # Save for export
            context.user_data["last_response"] = {
                "question": f"Поиск норматива {norm_code}",
                "answer": response_text,
                "store": "multistore",
                "timestamp": datetime.now().isoformat()
            }

            if len(response_text) > 4000:
                parts = [response_text[i:i+4000] for i in range(0, len(response_text), 4000)]
                await status_msg.edit_text(parts[0])
                for part in parts[1:]:
                    await update.message.reply_text(part)
                await update.message.reply_text("Export:", reply_markup=get_export_keyboard())
            else:
                await status_msg.edit_text(response_text, reply_markup=get_export_keyboard())
        else:
            await status_msg.edit_text(
                f"Норматив '{norm_code}' не найден ни в одном тендере."
            )
        return

    # Search in specific store
    status_msg = await update.message.reply_text(
        f"Поиск норматива '{norm_code}' в {store.get('name')}..."
    )

    try:
        norm_prompt = f"""Роль: инженер ПТО, проверяющий соответствие документации нормативным требованиям.

ЗАДАЧА: Найди ВСЕ упоминания норматива '{norm_code}' в загруженной документации.

ДЛЯ КАЖДОГО УПОМИНАНИЯ УКАЖИ:
1. **Документ**: название файла
2. **Раздел/страница**: где именно упоминается
3. **Контекст**: цитата (2-3 предложения вокруг упоминания)
4. **Требование**: что конкретно требуется по этому нормативу
5. **Статус**: Выполнимо / Требует уточнения / Противоречит другим требованиям

Если норматив НЕ найден:
- Укажи это явно
- Предложи, в каких разделах он ДОЛЖЕН быть упомянут
- Отметь это как потенциальный риск

ФОРМАТ ВЫВОДА:

## Норматив: {norm_code}

### Найдено упоминаний: X

| № | Документ | Раздел | Контекст | Требование | Статус |
|---|----------|--------|----------|------------|--------|
| 1 | ... | ... | ... | ... | ... |

### Выводы и рекомендации:
- ..."""

        answer = gemini_client.ask_question(
            store["id"],
            norm_prompt,
            model=GEMINI_MODEL_FLASH
        )

        if answer:
            # Save for export
            context.user_data["last_response"] = {
                "question": f"Поиск норматива {norm_code}",
                "answer": answer,
                "store": store.get("name", ""),
                "timestamp": datetime.now().isoformat()
            }

            if len(answer) > 4000:
                parts = [answer[i:i+4000] for i in range(0, len(answer), 4000)]
                await status_msg.edit_text(parts[0])
                for part in parts[1:]:
                    await update.message.reply_text(part)
                await update.message.reply_text("Export:", reply_markup=get_export_keyboard())
            else:
                await status_msg.edit_text(answer, reply_markup=get_export_keyboard())
        else:
            await status_msg.edit_text(
                f"Не удалось выполнить поиск норматива '{norm_code}'."
            )

    except Exception as e:
        logger.exception("Error in norm search")
        await status_msg.edit_text(f"Error: {str(e)[:500]}")


async def memory_cleanup_job(context: ContextTypes.DEFAULT_TYPE):
    """JobQueue callback for weekly memory cleanup"""
    logger.info("Running scheduled memory cleanup...")
    memory_client.cleanup_old_entries(days=MEMORY_CLEANUP_DAYS)
    stats = memory_client.get_stats()
    logger.info(f"Memory stats after cleanup: {stats}")


def main():
    """Start the bot"""
    if not BOT_TOKEN:
        print("Error: BOT_TOKEN not set in .env file")
        sys.exit(1)

    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not set in .env file")
        print("Get your API key from: https://aistudio.google.com/app/apikey")
        sys.exit(1)

    print("Starting Gemini Bot...")
    print(f"Stores: {len(gemini_client.stores) if gemini_client else 0}")
    print(f"Routing: {'enabled' if router else 'disabled'}")
    print(f"Model Flash (simple/medium): {GEMINI_MODEL_FLASH}")
    print(f"Model Pro (complex): {GEMINI_MODEL_PRO}")
    print("Smart model selection based on query complexity")

    app = Application.builder().token(BOT_TOKEN).build()

    # Schedule weekly memory cleanup (every Sunday at 4:00 AM)
    job_queue = app.job_queue
    job_queue.run_daily(
        memory_cleanup_job,
        time=time(hour=4, minute=0),
        days=(6,),  # Sunday
        name="memory_cleanup"
    )
    print("Scheduled: Weekly memory cleanup (Sundays 4:00 AM)")

    # Schedule daily auto-sync (3:00 AM)
    job_queue.run_daily(
        auto_sync_callback,
        time=time(hour=3, minute=0),
        name="auto_sync"
    )
    print("Scheduled: Daily auto-sync (3:00 AM)")

    # Add handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", start))
    app.add_handler(CommandHandler("add", add_store))
    app.add_handler(CommandHandler("addstore", add_store))
    app.add_handler(CommandHandler("upload", upload_file))
    app.add_handler(CommandHandler("uploadurl", upload_from_url))
    app.add_handler(CommandHandler("list", list_stores))
    app.add_handler(CommandHandler("stores", list_stores))
    app.add_handler(CommandHandler("select", select_store))
    app.add_handler(CommandHandler("status", check_status))
    app.add_handler(CommandHandler("sync", sync_stores))
    app.add_handler(CommandHandler("delete", delete_store))
    app.add_handler(CommandHandler("deletestore", delete_store))
    app.add_handler(CommandHandler("rename", rename_store))
    app.add_handler(CommandHandler("think", handle_think))
    app.add_handler(CommandHandler("clear", clear_memory))
    app.add_handler(CommandHandler("compare", compare_stores))
    app.add_handler(CommandHandler("setsync", set_sync))
    app.add_handler(CommandHandler("syncnow", sync_now))
    app.add_handler(CommandHandler("export", export_response))

    # Phase 1 Quick Wins - new commands
    app.add_handler(CommandHandler("summary", handle_summary))
    app.add_handler(CommandHandler("generate_rfi", handle_generate_rfi))
    app.add_handler(CommandHandler("rfi", handle_generate_rfi))  # alias
    app.add_handler(CommandHandler("norm", handle_norm))

    # Callback handler for export buttons
    app.add_handler(CallbackQueryHandler(handle_export_callback, pattern="^export_"))

    # File handler (for /upload flow)
    app.add_handler(MessageHandler(filters.Document.ALL, handle_file))

    # Photo handler
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Voice handler
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    # Question handler
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

    # Start polling
    print("Bot is running! Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
