#!/usr/bin/env python3
"""
NotebookLM Telegram Bot with Smart Routing

Sends questions to NotebookLM with intelligent notebook selection via Claude API.
"""

import asyncio
import subprocess
import logging
import sys
import json
import os
import shlex
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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
    ANTHROPIC_API_KEY,
    SKILL_PATH,
    SCRIPTS_PATH,
    SKILL_PYTHON,
    ALLOWED_USERS,
    QUERY_TIMEOUT,
    MAX_PARALLEL_QUERIES,
    AUTO_SYNC_INTERVAL,
    ADMIN_USER_ID,
    NOTEBOOKS_FILE,
)
from router import NotebookRouter, add_notebook_to_library, generate_descriptions
from notebooks_manager import (
    extract_notebook_id,
    add_notebook as add_notebook_to_file,
    list_notebooks as get_all_notebooks,
    load_notebooks,
)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Initialize router - use new notebooks.json format
router = None
if ANTHROPIC_API_KEY:
    router = NotebookRouter(ANTHROPIC_API_KEY, NOTEBOOKS_FILE)
    logger.info("Smart routing enabled with Claude API")
else:
    logger.warning("ANTHROPIC_API_KEY not set - smart routing disabled")

# Thread pool for parallel queries
executor = ThreadPoolExecutor(max_workers=MAX_PARALLEL_QUERIES)


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

    routing_status = "enabled" if router else "disabled (no API key)"
    admin_note = " (you are admin)" if is_admin(update.effective_user.id) else ""

    await update.message.reply_text(
        f"NotebookLM Router Bot{admin_note}\n\n"
        f"Smart routing: {routing_status}\n\n"
        "Commands:\n"
        "/list - Show all notebooks\n"
        "/status - Check status\n\n"
        "Just send a message to query your notebooks!\n"
        "The bot will automatically find the right notebook."
    )


def fetch_notebook_info(url: str) -> dict:
    """Fetch notebook name and description from NotebookLM page."""
    try:
        result = subprocess.run(
            [
                str(SKILL_PYTHON),
                "-c",
                f'''
import sys
import json
import time
sys.path.insert(0, str({repr(str(SCRIPTS_PATH))}))
from patchright.sync_api import sync_playwright
from browser_utils import BrowserFactory

with sync_playwright() as playwright:
    context = BrowserFactory.launch_persistent_context(playwright, headless=True)
    page = context.pages[0] if context.pages else context.new_page()

    try:
        page.goto("{url}", wait_until="domcontentloaded", timeout=30000)
        time.sleep(3)

        # Extract notebook name from title or header
        name = ""
        name_selectors = [
            'input[aria-label*="title"]',
            '.notebook-title',
            'h1',
            '[data-notebook-title]',
        ]
        for sel in name_selectors:
            try:
                el = page.query_selector(sel)
                if el:
                    val = el.get_attribute("value") or el.inner_text()
                    if val and val.strip():
                        name = val.strip()
                        break
            except:
                pass

        if not name:
            # Fallback to page title
            title = page.title()
            if title and "NotebookLM" in title:
                name = title.replace(" - NotebookLM", "").strip()
            else:
                name = title or "Untitled"

        # Extract description from sources
        description = ""
        desc_selectors = [
            '.source-card .source-title',
            '.sources-list .source-name',
            '[data-source-title]',
        ]
        sources = []
        for sel in desc_selectors:
            try:
                elements = page.query_selector_all(sel)
                for el in elements:
                    text = el.inner_text().strip()
                    if text and text not in sources:
                        sources.append(text)
            except:
                pass

        if sources:
            description = "Sources: " + ", ".join(sources[:5])

        print(json.dumps({{"name": name, "description": description}}))
    finally:
        context.close()
'''
            ],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=60,
            cwd=str(SKILL_PATH),
            env={"PYTHONIOENCODING": "utf-8", **os.environ},
        )

        if result.returncode == 0 and result.stdout.strip():
            # Find JSON in output
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if line.startswith('{'):
                    return json.loads(line)

        return {"name": "", "description": ""}
    except Exception as e:
        logger.error(f"Error fetching notebook info: {e}")
        return {"name": "", "description": ""}


async def add_notebook(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /add command - add notebook to library (admin only)"""
    user_id = update.effective_user.id

    if not check_user_allowed(user_id):
        await update.message.reply_text("Access denied.")
        return

    # Check if user is admin
    if not is_admin(user_id):
        await update.message.reply_text("Only admin can add notebooks.")
        return

    # Get URL from command arguments
    message_text = update.message.text
    args_text = re.sub(r'^/add\s*', '', message_text, flags=re.IGNORECASE).strip()

    if not args_text:
        await update.message.reply_text(
            "Usage: /add <notebook_url>\n\n"
            "Example:\n"
            "/add https://notebooklm.google.com/notebook/8b1d2368-449b-4bfc-abfa-5465b3a27981"
        )
        return

    url = args_text.split()[0]

    # Validate URL and extract ID
    notebook_id = extract_notebook_id(url)
    if not notebook_id:
        await update.message.reply_text(
            "Invalid NotebookLM URL.\n"
            "Expected format: https://notebooklm.google.com/notebook/{id}"
        )
        return

    # Store URL and start the add flow - ask for name
    context.user_data["pending_add"] = {
        "url": url,
        "id": notebook_id,
        "step": "waiting_name"
    }

    await update.message.reply_text(
        f"Notebook ID: {notebook_id}\n\n"
        "Enter notebook name:"
    )


async def handle_pending_add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle step-by-step notebook addition flow"""
    pending = context.user_data.get("pending_add")
    if not pending:
        return False

    text = update.message.text.strip()
    step = pending.get("step")

    # Step 1: Waiting for name
    if step == "waiting_name":
        pending["name"] = text
        pending["step"] = "waiting_description"
        context.user_data["pending_add"] = pending

        await update.message.reply_text(
            f"Name: {text}\n\n"
            "Now enter notebook description:"
        )
        return True

    # Step 2: Waiting for description
    if step == "waiting_description":
        description = text

        # Add notebook to file
        success, message = add_notebook_to_file(
            NOTEBOOKS_FILE,
            pending["url"],
            pending["name"],
            description
        )

        # Clear pending
        del context.user_data["pending_add"]

        if success:
            # Reload router library
            if router:
                router.reload_library()

        await update.message.reply_text(message)
        return True

    return False


async def list_notebooks_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /list command - show all notebooks"""
    if not check_user_allowed(update.effective_user.id):
        await update.message.reply_text("Access denied.")
        return

    notebooks = get_all_notebooks(NOTEBOOKS_FILE)

    if not notebooks:
        await update.message.reply_text("No notebooks yet.\nAdmin can add with /add command.")
        return

    # Simple numbered list of names with descriptions
    text = f"Notebooks ({len(notebooks)}):\n\n"
    for i, nb in enumerate(notebooks, 1):
        name = nb.get("name", "Unnamed")
        desc = nb.get("description", "")
        text += f"{i}. {name}\n"
        if desc:
            text += f"   {desc[:50]}{'...' if len(desc) > 50 else ''}\n"

    await update.message.reply_text(text)


async def reload_library(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /reload command"""
    if not check_user_allowed(update.effective_user.id):
        await update.message.reply_text("Access denied.")
        return

    if router:
        router.reload_library()
        await update.message.reply_text(f"Library reloaded. {len(router.notebooks)} notebooks.")
    else:
        await update.message.reply_text("Router not initialized.")


async def check_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command"""
    if not check_user_allowed(update.effective_user.id):
        await update.message.reply_text("Access denied.")
        return

    status_lines = [
        "Status:",
        f"- Smart routing: {'Yes' if router else 'No (no API key)'}",
        f"- Notebooks loaded: {len(router.notebooks) if router else 0}",
        f"- Skill path: {SKILL_PATH}",
    ]

    # Check auth
    try:
        result = subprocess.run(
            [str(SKILL_PYTHON), str(SCRIPTS_PATH / "auth_manager.py"), "status"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=30,
            cwd=str(SKILL_PATH),
            env={"PYTHONIOENCODING": "utf-8", **os.environ},
        )
        if "Yes" in result.stdout or "authenticated" in result.stdout.lower():
            status_lines.append("- NotebookLM auth: OK")
        else:
            status_lines.append("- NotebookLM auth: Not authenticated")
    except Exception as e:
        status_lines.append(f"- NotebookLM auth: Error ({e})")

    await update.message.reply_text("\n".join(status_lines))


async def sync_notebooks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /sync command - sync notebooks from NotebookLM account"""
    if not check_user_allowed(update.effective_user.id):
        await update.message.reply_text("Access denied.")
        return

    status_msg = await update.message.reply_text(
        "Syncing notebooks from your NotebookLM account...\n"
        "Fetching descriptions from each notebook.\n"
        "This may take 2-5 minutes depending on notebook count."
    )

    try:
        # Run sync script (visible for better reliability)
        result = subprocess.run(
            [str(SKILL_PYTHON), str(SCRIPTS_PATH / "sync_notebooks.py"), "--visible"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=600,  # 10 min max
            cwd=str(SKILL_PATH),
            env={"PYTHONIOENCODING": "utf-8", **os.environ},
        )

        output = result.stdout + result.stderr

        # Count notebooks found
        import re
        found_match = re.search(r'Total notebooks synced: (\d+)', output)
        added_match = re.search(r'Added (\d+) new notebooks', output)
        updated_match = re.search(r'updated (\d+) descriptions', output)

        found = int(found_match.group(1)) if found_match else 0
        added = int(added_match.group(1)) if added_match else 0
        updated = int(updated_match.group(1)) if updated_match else 0

        if found == 0:
            await status_msg.edit_text(
                "No notebooks found.\n"
                "Make sure you're authenticated (run /status to check)."
            )
            return

        # Check how many have descriptions
        library_path = SKILL_PATH / "data" / "library.json"
        missing_desc = 0
        if library_path.exists():
            with open(library_path, "r", encoding="utf-8") as f:
                lib = json.load(f)
                for nb in lib.get("notebooks", []):
                    if not nb.get("description"):
                        missing_desc += 1

        # Generate descriptions via GPT only for those without
        if missing_desc > 0 and ANTHROPIC_API_KEY:
            await status_msg.edit_text(
                f"Found {found} notebooks.\n"
                f"Generating descriptions for {missing_desc} without them..."
            )
            descriptions_added = generate_descriptions(ANTHROPIC_API_KEY, library_path)
        else:
            descriptions_added = 0

        # Reload router
        if router:
            router.reload_library()

        await status_msg.edit_text(
            f"Sync complete!\n"
            f"- Notebooks: {found}\n"
            f"- New: {added}\n"
            f"- Descriptions from NotebookLM: {found - missing_desc}\n"
            f"- Descriptions via GPT: {descriptions_added}\n\n"
            f"Use /list to see all notebooks."
        )

    except subprocess.TimeoutExpired:
        await status_msg.edit_text("Sync timed out. Try /sync again or use fewer notebooks.")
    except Exception as e:
        logger.exception("Sync error")
        await status_msg.edit_text(f"Error: {str(e)[:500]}")


def query_notebook_sync(url: str, question: str) -> dict:
    """
    Query a single notebook (sync, for thread pool).

    Returns dict with 'url', 'answer', 'error'
    """
    try:
        result = subprocess.run(
            [
                str(SKILL_PYTHON),
                str(SCRIPTS_PATH / "ask_question.py"),
                "--notebook-url", url,
                "--question", question,
                "--show-browser",
            ],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=QUERY_TIMEOUT,
            cwd=str(SKILL_PATH),
            env={"PYTHONIOENCODING": "utf-8", **os.environ},
        )

        if result.returncode != 0:
            return {"url": url, "answer": None, "error": result.stderr[:500]}

        answer = extract_answer(result.stdout)
        return {"url": url, "answer": answer, "error": None}

    except subprocess.TimeoutExpired:
        return {"url": url, "answer": None, "error": "Timeout"}
    except Exception as e:
        return {"url": url, "answer": None, "error": str(e)}


def extract_answer(output: str) -> str:
    """Extract clean answer from ask_question.py output"""
    lines = output.split("\n")
    answer_lines = []
    in_answer = False

    for line in lines:
        # Skip status/progress lines
        if any(x in line for x in [
            "🔐", "✅", "❌", "⏳", "💾", "🔍", "🌐", "📚", "💬",
            "Opening", "Waiting", "Found input", "Typing", "Submitting",
            "Got answer", "===", "---", "Asking:"
        ]):
            continue

        # Start capturing after "Question:" line
        if line.strip().startswith("Question:"):
            in_answer = True
            continue

        # Stop at "EXTREMELY IMPORTANT" footer
        if "EXTREMELY IMPORTANT" in line:
            break

        if in_answer and line.strip():
            answer_lines.append(line)

    # Join and clean up
    answer = "\n".join(answer_lines).strip()

    # Remove any trailing "EXTREMELY IMPORTANT..." text that might have been included
    if "EXTREMELY IMPORTANT" in answer:
        answer = answer.split("EXTREMELY IMPORTANT")[0].strip()

    # If no answer found, try to get any meaningful text
    if not answer:
        meaningful = [
            l.strip() for l in lines
            if l.strip()
            and not any(x in l for x in [
                "🔐", "✅", "❌", "⏳", "💾", "🔍", "🌐", "📚", "💬",
                "Opening", "Waiting", "Found", "Typing", "Submitting",
                "===", "---", "Question:", "EXTREMELY IMPORTANT"
            ])
        ]
        answer = "\n".join(meaningful)

    return answer


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user questions with smart routing"""
    user_id = update.effective_user.id

    if not check_user_allowed(user_id):
        await update.message.reply_text("Access denied.")
        return

    # Check if this is part of pending /add flow
    if await handle_pending_add(update, context):
        return

    question = update.message.text.strip()
    if not question:
        return

    # Check if we have notebooks
    if router and not router.notebooks:
        await update.message.reply_text(
            "No notebooks in library.\n"
            "Use /add to add notebooks first."
        )
        return

    if not router:
        await update.message.reply_text(
            "Smart routing not available (no API key).\n"
            "Set ANTHROPIC_API_KEY in .env"
        )
        return

    # Send typing indicator
    await update.message.chat.send_action("typing")

    # Route the question
    status_msg = await update.message.reply_text("Analyzing question...")

    try:
        selected, reasoning = router.route_with_reasoning(question, max_notebooks=3)

        if not selected:
            await status_msg.edit_text("Could not find relevant notebooks.")
            return

        # Show routing decision
        notebook_names = [nb.get("name", "?") for nb in selected]
        await status_msg.edit_text(
            f"Selected: {', '.join(notebook_names)}\n"
            f"Reason: {reasoning}\n\n"
            f"Querying {len(selected)} notebook(s)..."
        )

        # Query notebooks sequentially (to avoid browser profile conflicts)
        loop = asyncio.get_event_loop()
        answers = []

        for nb in selected:
            nb_name = nb.get("name", "Unknown")
            nb_url = nb.get("url")

            # Enhance the prompt for this specific notebook
            enhanced_question = router.enhance_prompt(question, nb)
            
            await status_msg.edit_text(
                f"📚 Querying: {nb_name}...\n"
                f"💬 Enhanced: {enhanced_question[:100]}..."
            )

            result = await loop.run_in_executor(
                executor,
                query_notebook_sync,
                nb_url,
                enhanced_question  # Use enhanced question
            )

            if result["answer"]:
                answers.append({
                    "notebook": nb_name,
                    "answer": result["answer"]
                })
                # Got an answer - stop querying more notebooks
                break
            elif result["error"]:
                logger.warning(f"Query error for {nb_name}: {result['error']}")

        if not answers:
            await status_msg.edit_text(
                f"Selected: {', '.join(notebook_names)}\n\n"
                "No answers received. Notebooks may be unavailable."
            )
            return

        # Format response - just the answer, clean and simple
        if len(answers) == 1:
            response = answers[0]['answer']
        else:
            # Multiple answers - add source headers
            response = ""
            for ans in answers:
                response += f"[{ans['notebook']}]\n{ans['answer']}\n\n"
            response = response.strip()

        # Split long messages
        if len(response) > 4000:
            parts = [response[i:i+4000] for i in range(0, len(response), 4000)]
            await status_msg.edit_text(parts[0])
            for part in parts[1:]:
                await update.message.reply_text(part)
        else:
            await status_msg.edit_text(response)

    except Exception as e:
        logger.exception("Error handling question")
        await status_msg.edit_text(f"Error: {str(e)[:500]}")




async def background_sync(context: ContextTypes.DEFAULT_TYPE):
    """Background job to sync notebooks periodically"""
    logger.info("Running background sync...")

    try:
        # Run quick sync (no descriptions, just list)
        result = subprocess.run(
            [str(SKILL_PYTHON), str(SCRIPTS_PATH / "sync_notebooks.py"),
             "--headless", "--no-descriptions"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=120,
            cwd=str(SKILL_PATH),
            env={"PYTHONIOENCODING": "utf-8", **os.environ},
        )

        # Reload router
        if router:
            router.reload_library()
            logger.info(f"Background sync complete. {len(router.notebooks)} notebooks.")

    except Exception as e:
        logger.error(f"Background sync error: {e}")


async def startup_sync(app):
    """Run quick sync on bot startup (just reload existing library, don't re-fetch)"""
    logger.info("Running startup sync...")

    try:
        # Just reload existing library on startup (fast)
        # Full sync happens every 30 min via background_sync
        if router:
            router.reload_library()
            logger.info(f"Startup sync complete. {len(router.notebooks)} notebooks loaded.")

    except Exception as e:
        logger.error(f"Startup sync error: {e}")


def main():
    """Start the bot"""
    if not BOT_TOKEN:
        print("Error: BOT_TOKEN not set in .env file")
        sys.exit(1)

    if not SKILL_PATH.exists():
        print(f"Error: NotebookLM skill not found at {SKILL_PATH}")
        sys.exit(1)

    if not SKILL_PYTHON.exists():
        print(f"Error: Python venv not found at {SKILL_PYTHON}")
        print("Run setup in notebooklm-skill first")
        sys.exit(1)

    if not ANTHROPIC_API_KEY:
        print("Warning: ANTHROPIC_API_KEY not set - smart routing disabled")

    print("Starting NotebookLM Router Bot...")
    print(f"Skill path: {SKILL_PATH}")
    print(f"Smart routing: {'enabled' if router else 'disabled'}")
    if router:
        print(f"Notebooks loaded: {len(router.notebooks)}")

    # Create application
    app = Application.builder().token(BOT_TOKEN).build()

    # Add handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", start))
    app.add_handler(CommandHandler("add", add_notebook))
    app.add_handler(CommandHandler("list", list_notebooks_cmd))
    app.add_handler(CommandHandler("reload", reload_library))
    app.add_handler(CommandHandler("status", check_status))
    app.add_handler(CommandHandler("sync", sync_notebooks))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

    # Schedule background sync
    if AUTO_SYNC_INTERVAL > 0:
        print(f"Auto-sync enabled: every {AUTO_SYNC_INTERVAL} minutes")
        app.job_queue.run_repeating(
            background_sync,
            interval=AUTO_SYNC_INTERVAL * 60,  # Convert to seconds
            first=AUTO_SYNC_INTERVAL * 60,  # First run after interval
            name="background_sync"
        )

    # Run startup sync
    app.post_init = startup_sync

    # Start polling
    print("Bot is running! Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
