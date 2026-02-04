import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class UserStateClient:
    """Persist per-user state like selected store."""

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.state: Dict = {}
        self._load_state()

    def _load_state(self):
        if not self.state_file.exists():
            self.state = {}
            return

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                self.state = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load user state: {e}")
            self.state = {}

    def _save_state(self):
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save user state: {e}")

    def set_selected_store(self, user_id: int, store_id: str, store_name: str):
        user_key = str(user_id)
        self.state[user_key] = {
            "selected_store_id": store_id,
            "selected_store_name": store_name,
            "updated_at": datetime.now().isoformat()
        }
        self._save_state()

    def clear_selected_store(self, user_id: int):
        user_key = str(user_id)
        if user_key in self.state:
            del self.state[user_key]
            self._save_state()

    def get_selected_store(self, user_id: int) -> Optional[Dict]:
        user_key = str(user_id)
        return self.state.get(user_key)

    def clear_store_for_all(self, store_id: str):
        if not store_id:
            return

        to_remove = []
        for user_key, data in self.state.items():
            if data.get("selected_store_id") == store_id:
                to_remove.append(user_key)

        for user_key in to_remove:
            del self.state[user_key]

        if to_remove:
            self._save_state()
