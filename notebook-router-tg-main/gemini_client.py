"""
Gemini 3 Flash File Search Client

Full Gemini 3-powered document Q&A.
Uses thinking_level for reasoning control.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Literal

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

ThinkingLevel = Literal["minimal", "low", "medium", "high"]


class GeminiFileSearchClient:
    """
    Client for Gemini File Search API with Gemini 3 thinking capabilities.

    Manages file search stores and answers questions
    with configurable thinking level.
    """

    def __init__(self, api_key: str, stores_file: Path):
        """
        Initialize the client.

        Args:
            api_key: Gemini API key
            stores_file: Path to JSON file storing store metadata
        """
        self.client = genai.Client(api_key=api_key)
        self.stores_file = stores_file
        self.stores: List[Dict] = []
        self._load_stores()

    def _load_stores(self):
        """Load stores metadata from file."""
        if not self.stores_file.exists():
            self.stores = []
            return

        try:
            with open(self.stores_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.stores = data if isinstance(data, list) else []
                logger.info(f"Loaded {len(self.stores)} stores")
        except Exception as e:
            logger.error(f"Failed to load stores: {e}")
            self.stores = []

    def _save_stores(self):
        """Save stores metadata to file."""
        try:
            with open(self.stores_file, "w", encoding="utf-8") as f:
                json.dump(self.stores, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save stores: {e}")

    def reload_stores(self):
        """Reload stores from disk."""
        self._load_stores()

    def create_store(self, name: str, description: str = "") -> Optional[Dict]:
        """
        Create a new file search store.

        Args:
            name: Display name for the store
            description: Description for routing purposes

        Returns:
            Store metadata dict or None on failure
        """
        try:
            file_search_store = self.client.file_search_stores.create(
                config={'display_name': name}
            )

            store_data = {
                "id": file_search_store.name,
                "name": name,
                "description": description,
                "documents": []
            }

            self.stores.append(store_data)
            self._save_stores()

            logger.info(f"Created store: {name} ({file_search_store.name})")
            return store_data

        except Exception as e:
            logger.error(f"Failed to create store: {e}")
            return None

    def upload_file(
        self,
        store_id: str,
        file_path: Path,
        display_name: str = "",
        wait: bool = True,
        timeout: int = 300
    ) -> bool:
        """
        Upload a file to a store.

        Args:
            store_id: Store resource name
            file_path: Path to file to upload
            display_name: Display name for the file
            wait: Wait for processing to complete
            timeout: Max seconds to wait

        Returns:
            True if successful
        """
        try:
            if not display_name:
                display_name = file_path.name

            operation = self.client.file_search_stores.upload_to_file_search_store(
                file=str(file_path),
                file_search_store_name=store_id,
                config={'display_name': display_name}
            )

            if wait:
                start_time = time.time()
                while not operation.done:
                    if time.time() - start_time > timeout:
                        logger.error(f"Upload timeout for {file_path}")
                        return False
                    time.sleep(5)
                    operation = self.client.operations.get(operation)

            for store in self.stores:
                if store["id"] == store_id:
                    store["documents"].append({
                        "name": display_name,
                        "path": str(file_path)
                    })
                    break
            self._save_stores()

            logger.info(f"Uploaded {file_path} to {store_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            return False

    def ask_question(
        self,
        store_id: str,
        question: str,
        model: str = "gemini-3-flash-preview",
        thinking_level: Optional[ThinkingLevel] = None
    ) -> Optional[str]:
        """
        Ask a question to a specific store.

        Args:
            store_id: Store resource name
            question: User's question
            model: Gemini model to use
            thinking_level: Thinking level (minimal, low, medium, high)

        Returns:
            Answer text or None on failure
        """
        try:
            config_params = {
                "tools": [
                    types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_names=[store_id]
                        )
                    )
                ]
            }

            # Add thinking config for Gemini 3
            if thinking_level:
                config_params["thinking_config"] = types.ThinkingConfig(
                    thinking_level=thinking_level
                )

            response = self.client.models.generate_content(
                model=model,
                contents=question,
                config=types.GenerateContentConfig(**config_params)
            )

            if response and response.text:
                logger.info(f"Got answer for question: {question[:50]}...")
                return response.text

            return None

        except Exception as e:
            logger.error(f"Failed to get answer: {e}")
            return None

    def ask_with_thinking(
        self,
        store_id: str,
        question: str,
        thinking_level: ThinkingLevel = "high",
        model: str = "gemini-3-flash-preview"
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Ask a question with thinking mode enabled.
        Returns both the thinking process and the final answer.

        Args:
            store_id: Store resource name
            question: User's question
            thinking_level: Level of thinking (minimal, low, medium, high)
            model: Gemini model to use

        Returns:
            Tuple of (answer, thinking_text) or (None, None) on failure
        """
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=question,
                config=types.GenerateContentConfig(
                    tools=[
                        types.Tool(
                            file_search=types.FileSearch(
                                file_search_store_names=[store_id]
                            )
                        )
                    ],
                    thinking_config=types.ThinkingConfig(
                        thinking_level=thinking_level
                    )
                )
            )

            if not response or not response.candidates:
                return None, None

            thinking_text = None
            answer_text = None

            for part in response.candidates[0].content.parts:
                if hasattr(part, 'thought') and part.thought:
                    thinking_text = part.text
                elif hasattr(part, 'text'):
                    answer_text = part.text

            logger.info(f"Got thinking answer for: {question[:50]}...")
            return answer_text, thinking_text

        except Exception as e:
            logger.error(f"Failed to get thinking answer: {e}")
            return None, None

    def get_store_by_name(self, name: str) -> Optional[Dict]:
        """Find store by display name."""
        name_lower = name.lower()
        for store in self.stores:
            if store.get("name", "").lower() == name_lower:
                return store
        return None

    def get_store_by_id(self, store_id: str) -> Optional[Dict]:
        """Find store by ID."""
        for store in self.stores:
            if store.get("id") == store_id:
                return store
        return None

    def list_stores(self) -> List[Dict]:
        """Get all stores."""
        return self.stores

    def delete_store(self, store_id: str) -> bool:
        """
        Delete a store.

        Args:
            store_id: Store resource name

        Returns:
            True if successful
        """
        try:
            self.client.file_search_stores.delete(name=store_id)

            self.stores = [s for s in self.stores if s.get("id") != store_id]
            self._save_stores()

            logger.info(f"Deleted store: {store_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete store: {e}")
            return False

    def sync_with_api(self):
        """
        Sync local stores list with API.
        Adds any stores that exist in API but not locally.
        """
        try:
            api_stores = list(self.client.file_search_stores.list())

            local_ids = {s.get("id") for s in self.stores}

            for api_store in api_stores:
                if api_store.name not in local_ids:
                    self.stores.append({
                        "id": api_store.name,
                        "name": api_store.display_name or "Unnamed",
                        "description": "",
                        "documents": []
                    })

            self._save_stores()
            logger.info(f"Synced stores. Total: {len(self.stores)}")

        except Exception as e:
            logger.error(f"Failed to sync stores: {e}")


# CLI for testing
if __name__ == "__main__":
    import sys
    import os
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set")
        sys.exit(1)

    stores_file = Path(__file__).parent / "stores.json"
    client = GeminiFileSearchClient(api_key, stores_file)

    if len(sys.argv) < 2:
        print("Gemini 3 Flash File Search CLI")
        print()
        print("Usage:")
        print("  python gemini_client.py create <name> [description]")
        print("  python gemini_client.py upload <store_name> <file_path>")
        print("  python gemini_client.py ask <store_name> <question>")
        print("  python gemini_client.py think <store_name> <question>  # with high thinking")
        print("  python gemini_client.py list")
        print("  python gemini_client.py sync")
        print("  python gemini_client.py delete <store_name>")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "create" and len(sys.argv) >= 3:
        name = sys.argv[2]
        desc = sys.argv[3] if len(sys.argv) > 3 else ""
        result = client.create_store(name, desc)
        if result:
            print(f"Created store: {result['name']} (ID: {result['id']})")
        else:
            print("Failed to create store")

    elif cmd == "upload" and len(sys.argv) >= 4:
        store_name = sys.argv[2]
        file_path = Path(sys.argv[3])
        store = client.get_store_by_name(store_name)
        if not store:
            print(f"Store not found: {store_name}")
            sys.exit(1)
        if not file_path.exists():
            print(f"File not found: {file_path}")
            sys.exit(1)
        print(f"Uploading {file_path}...")
        if client.upload_file(store["id"], file_path):
            print("Upload successful")
        else:
            print("Upload failed")

    elif cmd == "ask" and len(sys.argv) >= 4:
        store_name = sys.argv[2]
        question = " ".join(sys.argv[3:])
        store = client.get_store_by_name(store_name)
        if not store:
            print(f"Store not found: {store_name}")
            sys.exit(1)
        print(f"Asking: {question}")
        answer = client.ask_question(store["id"], question)
        if answer:
            print(f"\nAnswer:\n{answer}")
        else:
            print("No answer received")

    elif cmd == "think" and len(sys.argv) >= 4:
        store_name = sys.argv[2]
        question = " ".join(sys.argv[3:])
        store = client.get_store_by_name(store_name)
        if not store:
            print(f"Store not found: {store_name}")
            sys.exit(1)
        print(f"Thinking (high) about: {question}")
        answer, thinking = client.ask_with_thinking(store["id"], question, thinking_level="high")
        if thinking:
            print(f"\n--- Thinking Process ---\n{thinking}")
        if answer:
            print(f"\n--- Answer ---\n{answer}")
        else:
            print("No answer received")

    elif cmd == "list":
        stores = client.list_stores()
        if not stores:
            print("No stores found")
        else:
            print(f"Stores ({len(stores)}):")
            for i, s in enumerate(stores, 1):
                print(f"  {i}. {s['name']}")
                print(f"     ID: {s['id']}")
                if s.get('description'):
                    print(f"     Description: {s['description']}")
                if s.get('documents'):
                    print(f"     Documents: {len(s['documents'])}")

    elif cmd == "sync":
        client.sync_with_api()
        print(f"Synced. Total stores: {len(client.stores)}")

    elif cmd == "delete" and len(sys.argv) >= 3:
        store_name = sys.argv[2]
        store = client.get_store_by_name(store_name)
        if not store:
            print(f"Store not found: {store_name}")
            sys.exit(1)
        if client.delete_store(store["id"]):
            print(f"Deleted: {store_name}")
        else:
            print("Failed to delete")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
