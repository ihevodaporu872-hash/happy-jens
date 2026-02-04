"""
User Memory Client for Telegram Bot

Manages conversation history per user and store.
Supports automatic cleanup of old entries.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class UserMemoryClient:
    """
    Client for managing user conversation memory.

    Stores conversation history per user and store with automatic cleanup.
    """

    def __init__(self, memory_file: Path, max_messages: int = 5):
        """
        Initialize the memory client.

        Args:
            memory_file: Path to JSON file for storing memory
            max_messages: Maximum messages to keep per user/store pair
        """
        self.memory_file = memory_file
        self.max_messages = max_messages
        self.memory: Dict = {}
        self._load_memory()

    def _load_memory(self):
        """Load memory from file."""
        if not self.memory_file.exists():
            self.memory = {}
            return

        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                self.memory = json.load(f)
            logger.info(f"Loaded memory for {len(self.memory)} users")
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
            self.memory = {}

    def _save_memory(self):
        """Save memory to file."""
        try:
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")

    def add_message(
        self,
        user_id: int,
        store_id: str,
        role: str,
        content: str
    ):
        """
        Add a message to user's conversation history.

        Args:
            user_id: Telegram user ID
            store_id: Store ID or "global" for non-store messages
            role: Message role ("user" or "assistant")
            content: Message content
        """
        user_key = str(user_id)

        if user_key not in self.memory:
            self.memory[user_key] = {}

        if store_id not in self.memory[user_key]:
            self.memory[user_key][store_id] = {
                "messages": [],
                "last_interaction": None
            }

        store_memory = self.memory[user_key][store_id]

        # Add new message
        store_memory["messages"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

        # Keep only last N messages
        if len(store_memory["messages"]) > self.max_messages:
            store_memory["messages"] = store_memory["messages"][-self.max_messages:]

        store_memory["last_interaction"] = datetime.now().isoformat()

        self._save_memory()
        logger.debug(f"Added {role} message for user {user_id} in store {store_id}")

    def get_history(self, user_id: int, store_id: str) -> List[Dict]:
        """
        Get conversation history for user and store.

        Args:
            user_id: Telegram user ID
            store_id: Store ID

        Returns:
            List of message dicts with role, content, timestamp
        """
        user_key = str(user_id)

        if user_key not in self.memory:
            return []

        if store_id not in self.memory[user_key]:
            return []

        return self.memory[user_key][store_id].get("messages", [])

    def get_context_prompt(self, user_id: int, store_id: str) -> str:
        """
        Get formatted context prompt from conversation history.

        Args:
            user_id: Telegram user ID
            store_id: Store ID

        Returns:
            Formatted string with conversation history for prompt injection
        """
        history = self.get_history(user_id, store_id)

        if not history:
            return ""

        lines = ["Предыдущий контекст диалога:"]
        for msg in history:
            role_label = "Пользователь" if msg["role"] == "user" else "Ассистент"
            # Truncate long messages in context
            content = msg["content"]
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"{role_label}: {content}")

        lines.append("\nТекущий вопрос:")
        return "\n".join(lines)

    def clear_history(self, user_id: int, store_id: Optional[str] = None):
        """
        Clear conversation history for user.

        Args:
            user_id: Telegram user ID
            store_id: Optional store ID. If None, clears all stores for user.
        """
        user_key = str(user_id)

        if user_key not in self.memory:
            return

        if store_id:
            if store_id in self.memory[user_key]:
                del self.memory[user_key][store_id]
                logger.info(f"Cleared history for user {user_id} in store {store_id}")
        else:
            self.memory[user_key] = {}
            logger.info(f"Cleared all history for user {user_id}")

        self._save_memory()

    def cleanup_old_entries(self, days: int = 7):
        """
        Remove entries older than specified days.
        Called by JobQueue for automatic cleanup.

        Args:
            days: Remove entries older than this many days
        """
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.isoformat()

        cleaned_users = 0
        cleaned_stores = 0

        users_to_remove = []

        for user_key, user_data in self.memory.items():
            stores_to_remove = []

            for store_id, store_data in user_data.items():
                last_interaction = store_data.get("last_interaction")
                if last_interaction and last_interaction < cutoff_str:
                    stores_to_remove.append(store_id)

            for store_id in stores_to_remove:
                del user_data[store_id]
                cleaned_stores += 1

            if not user_data:
                users_to_remove.append(user_key)

        for user_key in users_to_remove:
            del self.memory[user_key]
            cleaned_users += 1

        if cleaned_stores > 0 or cleaned_users > 0:
            self._save_memory()
            logger.info(f"Memory cleanup: removed {cleaned_stores} store entries, {cleaned_users} empty users")

    def get_stats(self) -> Dict:
        """
        Get memory statistics.

        Returns:
            Dict with user count, total messages, etc.
        """
        total_users = len(self.memory)
        total_stores = 0
        total_messages = 0

        for user_data in self.memory.values():
            total_stores += len(user_data)
            for store_data in user_data.values():
                total_messages += len(store_data.get("messages", []))

        return {
            "users": total_users,
            "stores": total_stores,
            "messages": total_messages
        }
