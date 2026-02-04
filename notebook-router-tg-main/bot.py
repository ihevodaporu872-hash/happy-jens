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
from pathlib import Path

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
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
    GEMINI_MODEL,
    GEMINI_THINKING_LEVEL,
)
from router import NotebookRouter
from gemini_client import GeminiFileSearchClient

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


def check_user_allowed(user_id: int) -> bool:
    """Check if user is allowed to use the bot"""
    if not ALLOWED_USERS:
        return True
    return user_id in ALLOWED_USERS


def is_admin(user_id: int) -> bool:
    """Check if user is the admin"""
    return ADMIN_USER_ID is not None and user_id == ADMIN_USER_ID


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    if not check_user_allowed(update.effective_user.id):
        await update.message.reply_text("Access denied.")
        return

    gemini_status = "enabled" if gemini_client else "disabled (no API key)"
    routing_status = "enabled" if router else "disabled"
    admin_note = " (you are admin)" if is_admin(update.effective_user.id) else ""
    stores_count = len(gemini_client.stores) if gemini_client else 0

    await update.message.reply_text(
        f"Gemini 3 Flash Bot{admin_note}\n\n"
        f"Model: {GEMINI_MODEL}\n"
        f"File Search: {gemini_status}\n"
        f"Smart routing: {routing_status}\n"
        f"Stores: {stores_count}\n\n"
        "Commands:\n"
        "/list - Show all stores\n"
        "/status - Check status\n"
        "/think <question> - Deep thinking mode\n"
        "/add - Add new store (admin)\n"
        "/upload - Upload file (admin)\n\n"
        "Just send a message to query!"
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

    store = gemini_client.get_store_by_name(store_name)
    if not store:
        await update.message.reply_text(f"Store not found: {store_name}")
        return

    status_msg = await update.message.reply_text(f"Downloading file...")

    try:
        file = await document.get_file()
        temp_path = Path(f"/tmp/{document.file_name}")
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


async def check_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command"""
    if not check_user_allowed(update.effective_user.id):
        await update.message.reply_text("Access denied.")
        return

    status_lines = [
        "Status:",
        f"- Gemini API: {'OK' if gemini_client else 'Not configured'}",
        f"- Smart routing: {'OK' if router else 'Not configured'}",
        f"- Stores: {len(gemini_client.stores) if gemini_client else 0}",
        f"- Model: {GEMINI_MODEL}",
        f"- Thinking level: {GEMINI_THINKING_LEVEL}",
        "",
        "Powered by Gemini 3 Flash"
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

    store = gemini_client.get_store_by_name(args_text)
    if not store:
        await update.message.reply_text(f"Store not found: {args_text}")
        return

    if gemini_client.delete_store(store["id"]):
        if router:
            router.reload_library()
        await update.message.reply_text(f"Deleted: {args_text}")
    else:
        await update.message.reply_text("Failed to delete. Check logs.")


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


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user questions with smart routing"""
    user_id = update.effective_user.id

    if not check_user_allowed(user_id):
        await update.message.reply_text("Access denied.")
        return

    question = update.message.text.strip()
    if not question:
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

    status_msg = await update.message.reply_text("Analyzing question...")

    try:
        # Route the question
        if router and len(gemini_client.stores) > 1:
            selected, reasoning = router.route_with_reasoning(question, max_notebooks=1)
            if selected:
                store = selected[0]
                await status_msg.edit_text(
                    f"Selected: {store.get('name')}\n"
                    f"Reason: {reasoning}\n\n"
                    f"Getting answer..."
                )
            else:
                store = gemini_client.stores[0]
                await status_msg.edit_text(f"Querying {store.get('name')}...")
        else:
            store = gemini_client.stores[0]
            await status_msg.edit_text(f"Querying {store.get('name')}...")

        # Get answer from Gemini
        answer = gemini_client.ask_question(
            store["id"],
            question,
            model=GEMINI_MODEL
        )

        if answer:
            if len(answer) > 4000:
                parts = [answer[i:i+4000] for i in range(0, len(answer), 4000)]
                await status_msg.edit_text(parts[0])
                for part in parts[1:]:
                    await update.message.reply_text(part)
            else:
                await status_msg.edit_text(answer)
        else:
            await status_msg.edit_text(
                "No answer received.\n"
                "The store might be empty or the question couldn't be answered."
            )

    except Exception as e:
        logger.exception("Error handling question")
        await status_msg.edit_text(f"Error: {str(e)[:500]}")


def main():
    """Start the bot"""
    if not BOT_TOKEN:
        print("Error: BOT_TOKEN not set in .env file")
        sys.exit(1)

    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not set in .env file")
        print("Get your API key from: https://aistudio.google.com/app/apikey")
        sys.exit(1)

    print("Starting Gemini 3 Flash Bot...")
    print(f"Stores: {len(gemini_client.stores) if gemini_client else 0}")
    print(f"Routing: {'enabled' if router else 'disabled'}")
    print(f"Model: {GEMINI_MODEL}")
    print(f"Thinking level: {GEMINI_THINKING_LEVEL}")
    print("Powered by Gemini 3 Flash")

    app = Application.builder().token(BOT_TOKEN).build()

    # Add handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", start))
    app.add_handler(CommandHandler("add", add_store))
    app.add_handler(CommandHandler("upload", upload_file))
    app.add_handler(CommandHandler("list", list_stores))
    app.add_handler(CommandHandler("status", check_status))
    app.add_handler(CommandHandler("sync", sync_stores))
    app.add_handler(CommandHandler("delete", delete_store))
    app.add_handler(CommandHandler("think", handle_think))

    # File handler (for /upload flow)
    app.add_handler(MessageHandler(filters.Document.ALL, handle_file))

    # Question handler
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

    # Start polling
    print("Bot is running! Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
