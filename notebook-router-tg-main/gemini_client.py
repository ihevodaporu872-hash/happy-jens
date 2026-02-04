"""
Gemini 3 Flash File Search Client

Full Gemini 3-powered document Q&A.
Uses thinking_level for reasoning control.
"""

import json
import logging
import time
import concurrent.futures
from pathlib import Path
from typing import Optional, List, Dict, Literal

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

ThinkingLevel = Literal["minimal", "low", "medium", "high"]


# Note: Query type detection is now handled by QueryProcessor with ultrathinking
# Old pattern-based detection removed in favor of AI-powered understanding


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
        source_url: str = "",
        wait: bool = True,
        timeout: int = 300
    ) -> bool:
        """
        Upload a file to a store.

        Args:
            store_id: Store resource name
            file_path: Path to file to upload
            display_name: Display name for the file
            source_url: Source URL (Google Docs/Drive link)
            wait: Wait for processing to complete
            timeout: Max seconds to wait

        Returns:
            True if successful
        """
        try:
            from datetime import datetime

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
                    doc_entry = {
                        "name": display_name,
                        "path": str(file_path),
                        "uploaded_at": datetime.now().isoformat()
                    }
                    if source_url:
                        doc_entry["source_url"] = source_url
                    store["documents"].append(doc_entry)
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

    def analyze_image(
        self,
        image_path: Path,
        prompt: str = "Опиши что на изображении",
        model: str = "gemini-3-flash-preview"
    ) -> Optional[str]:
        """
        Analyze an image using Gemini Vision API.

        Args:
            image_path: Path to image file
            prompt: Prompt for analysis
            model: Gemini model to use

        Returns:
            Analysis text or None on failure
        """
        try:
            # Read image file
            with open(image_path, "rb") as f:
                image_data = f.read()

            # Determine MIME type
            suffix = image_path.suffix.lower()
            mime_types = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }
            mime_type = mime_types.get(suffix, "image/jpeg")

            # Create image part
            image_part = types.Part.from_bytes(data=image_data, mime_type=mime_type)

            # Send to Gemini
            response = self.client.models.generate_content(
                model=model,
                contents=[prompt, image_part]
            )

            if response and response.text:
                logger.info(f"Analyzed image: {image_path.name}")
                return response.text

            return None

        except Exception as e:
            logger.error(f"Failed to analyze image: {e}")
            return None

    def transcribe_voice(
        self,
        audio_path: Path,
        model: str = "gemini-3-flash-preview"
    ) -> Optional[str]:
        """
        Transcribe voice message using Gemini API.

        Args:
            audio_path: Path to audio file (OGG format from Telegram)
            model: Gemini model to use

        Returns:
            Transcribed text or None on failure
        """
        try:
            # Read audio file
            with open(audio_path, "rb") as f:
                audio_data = f.read()

            # Determine MIME type
            suffix = audio_path.suffix.lower()
            mime_types = {
                ".ogg": "audio/ogg",
                ".oga": "audio/ogg",
                ".mp3": "audio/mpeg",
                ".wav": "audio/wav",
                ".m4a": "audio/mp4",
            }
            mime_type = mime_types.get(suffix, "audio/ogg")

            # Create audio part
            audio_part = types.Part.from_bytes(data=audio_data, mime_type=mime_type)

            # Send to Gemini with transcription prompt
            response = self.client.models.generate_content(
                model=model,
                contents=[
                    "Расшифруй это голосовое сообщение на русском языке. "
                    "Верни только текст сообщения без дополнительных комментариев.",
                    audio_part
                ]
            )

            if response and response.text:
                logger.info(f"Transcribed voice: {audio_path.name}")
                return response.text.strip()

            return None

        except Exception as e:
            logger.error(f"Failed to transcribe voice: {e}")
            return None

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
                        "documents": [],
                        "sync_urls": [],
                        "last_sync": None,
                        "auto_sync_enabled": False
                    })

            self._save_stores()
            logger.info(f"Synced stores. Total: {len(self.stores)}")

        except Exception as e:
            logger.error(f"Failed to sync stores: {e}")

    def set_sync_urls(self, store_id: str, urls: List[str], auto_sync: bool = True) -> bool:
        """
        Set URLs for automatic synchronization of a store.

        Args:
            store_id: Store resource name
            urls: List of Google Drive/Docs URLs
            auto_sync: Enable automatic sync

        Returns:
            True if successful
        """
        for store in self.stores:
            if store["id"] == store_id:
                store["sync_urls"] = urls
                store["auto_sync_enabled"] = auto_sync
                self._save_stores()
                logger.info(f"Set sync URLs for {store.get('name')}: {len(urls)} URLs")
                return True
        return False

    def get_stores_for_sync(self) -> List[Dict]:
        """
        Get stores that need synchronization.

        Returns:
            List of stores with auto_sync_enabled=True and sync_urls set
        """
        return [
            store for store in self.stores
            if store.get("auto_sync_enabled") and store.get("sync_urls")
        ]

    def update_last_sync(self, store_id: str):
        """Update last_sync timestamp for a store."""
        from datetime import datetime
        for store in self.stores:
            if store["id"] == store_id:
                store["last_sync"] = datetime.now().isoformat()
                self._save_stores()
                break

    def ask_multistore_parallel(
        self,
        store_ids: List[str],
        question: str,
        model: str = "gemini-3-flash-preview",
        max_workers: int = 5
    ) -> List[Dict]:
        """
        Ask the same question to multiple stores in parallel.

        Args:
            store_ids: List of store resource names
            question: User's question
            model: Gemini model to use
            max_workers: Maximum parallel requests

        Returns:
            List of dicts with store_id, store_name, answer, has_result
        """
        results = []

        def query_store(store_id: str) -> Dict:
            store = self.get_store_by_id(store_id)
            store_name = store.get("name", "Unknown") if store else "Unknown"

            try:
                # Ask for a brief answer with key quote
                enhanced_question = (
                    f"{question}\n\n"
                    "Ответь кратко (1-2 предложения) и приведи ключевую цитату из документа, "
                    "если информация найдена. Если информация не найдена, скажи 'Не найдено'."
                )

                answer = self.ask_question(store_id, enhanced_question, model=model)

                if answer:
                    # Check if answer indicates no results
                    no_result_indicators = [
                        "не найден", "не содержит", "нет информации",
                        "отсутствует", "not found", "no information"
                    ]
                    has_result = not any(
                        ind in answer.lower() for ind in no_result_indicators
                    )
                else:
                    has_result = False
                    answer = "Ошибка запроса"

                return {
                    "store_id": store_id,
                    "store_name": store_name,
                    "answer": answer,
                    "has_result": has_result
                }

            except Exception as e:
                logger.error(f"Multistore query error for {store_id}: {e}")
                return {
                    "store_id": store_id,
                    "store_name": store_name,
                    "answer": f"Ошибка: {str(e)[:100]}",
                    "has_result": False
                }

        # Execute queries in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_store = {
                executor.submit(query_store, store_id): store_id
                for store_id in store_ids
            }

            for future in concurrent.futures.as_completed(future_to_store):
                result = future.result()
                results.append(result)

        # Sort results: stores with results first
        results.sort(key=lambda x: (not x["has_result"], x["store_name"]))

        logger.info(f"Multistore query complete: {len(results)} stores, "
                   f"{sum(1 for r in results if r['has_result'])} with results")

        return results

    def format_multistore_response(self, results: List[Dict]) -> str:
        """
        Format multistore query results for display.

        Args:
            results: Results from ask_multistore_parallel

        Returns:
            Formatted string for Telegram message
        """
        results_with_data = [r for r in results if r["has_result"]]

        if not results_with_data:
            return "Информация не найдена ни в одном из тендеров."

        lines = [f"Найдено в {len(results_with_data)} тендерах:\n"]

        for i, result in enumerate(results_with_data, 1):
            lines.append(f"{i}. **{result['store_name']}**")
            # Truncate long answers
            answer = result["answer"]
            if len(answer) > 500:
                answer = answer[:500] + "..."
            lines.append(f"   {answer}\n")

        return "\n".join(lines)

    def analyze_store_content(
        self,
        store_id: str,
        model: str = "gemini-3-pro-preview"
    ) -> Optional[Dict]:
        """
        Analyze store content to generate name and description.
        Uses Gemini Pro to understand documents and create summary.

        Args:
            store_id: Store resource name
            model: Gemini model (Pro recommended for analysis)

        Returns:
            Dict with 'name' and 'description' or None on failure
        """
        try:
            analysis_prompt = """Проанализируй документы в этой базе знаний и определи:

1. НАЗВАНИЕ ТЕНДЕРА/ПРОЕКТА - краткое официальное название (например: "ЖК Солнечный - Благоустройство", "Метро Кунцевская - Водопонижение")

2. КРАТКОЕ ОПИСАНИЕ - 2-3 предложения о сути проекта:
   - Что за объект
   - Какие основные работы
   - Ключевые особенности

Ответь СТРОГО в формате JSON:
{
    "name": "Краткое название тендера",
    "description": "Краткое описание проекта и основных работ"
}

Анализируй только факты из документов. JSON:"""

            response = self.client.models.generate_content(
                model=model,
                contents=analysis_prompt,
                config=types.GenerateContentConfig(
                    tools=[
                        types.Tool(
                            file_search=types.FileSearch(
                                file_search_store_names=[store_id]
                            )
                        )
                    ],
                    temperature=0.1,
                    max_output_tokens=500
                )
            )

            if not response or not response.text:
                return None

            # Parse JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', response.text)
            if json_match:
                data = json.loads(json_match.group())
                logger.info(f"Analyzed store: name='{data.get('name')}', desc='{data.get('description', '')[:50]}...'")
                return data

            return None

        except Exception as e:
            logger.error(f"Store analysis failed: {e}")
            return None

    def update_store_metadata(self, store_id: str, name: str = None, description: str = None) -> bool:
        """
        Update store name and description in local metadata.

        Args:
            store_id: Store resource name
            name: New name (optional)
            description: New description (optional)

        Returns:
            True if updated
        """
        for store in self.stores:
            if store["id"] == store_id:
                if name:
                    store["name"] = name
                if description:
                    store["description"] = description
                self._save_stores()
                logger.info(f"Updated store metadata: {name}")
                return True
        return False

    def ask_with_web_search(
        self,
        question: str,
        model: str = "gemini-3-flash-preview"
    ) -> Optional[str]:
        """
        Ask a question using Gemini with Google Search grounding.

        Args:
            question: User's question
            model: Gemini model to use

        Returns:
            Answer text with web search results or None on failure
        """
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=question,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                )
            )

            if response and response.text:
                logger.info(f"Got web search answer for: {question[:50]}...")
                return response.text

            return None

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return None

    def get_store_sources(self, store_id: str) -> List[Dict]:
        """
        Get list of documents with source URLs for a store.

        Args:
            store_id: Store resource name

        Returns:
            List of dicts with name and source_url
        """
        store = self.get_store_by_id(store_id)
        if not store:
            return []

        sources = []
        for doc in store.get("documents", []):
            if doc.get("source_url"):
                sources.append({
                    "name": doc.get("name", "Unknown"),
                    "source_url": doc.get("source_url")
                })
        return sources

    def format_sources_footer(self, store_id: str) -> str:
        """
        Format sources as footer text for answer.

        Args:
            store_id: Store resource name

        Returns:
            Formatted sources text or empty string
        """
        sources = self.get_store_sources(store_id)
        if not sources:
            return ""

        lines = ["\n\nИсточники:"]
        for src in sources[:5]:  # Limit to 5 sources
            lines.append(f"- {src['name']}: {src['source_url']}")

        return "\n".join(lines)

    def ask_with_sources(
        self,
        store_id: str,
        question: str,
        model: str = "gemini-3-flash-preview",
        thinking_level: Optional[ThinkingLevel] = None
    ) -> Optional[str]:
        """
        Ask a question and append source links if available.

        Args:
            store_id: Store resource name
            question: User's question
            model: Gemini model to use
            thinking_level: Thinking level

        Returns:
            Answer text with sources or None
        """
        answer = self.ask_question(store_id, question, model, thinking_level)
        if not answer:
            return None

        sources_footer = self.format_sources_footer(store_id)
        return answer + sources_footer

    def compare_stores(
        self,
        store_id_1: str,
        store_id_2: str,
        topic: str,
        model: str = "gemini-3-flash-preview"
    ) -> Optional[str]:
        """
        Compare information about a topic between two stores.

        Args:
            store_id_1: First store resource name
            store_id_2: Second store resource name
            topic: Topic to compare
            model: Gemini model to use

        Returns:
            Comparison text or None on failure
        """
        store_1 = self.get_store_by_id(store_id_1)
        store_2 = self.get_store_by_id(store_id_2)

        name_1 = store_1.get("name", "Store 1") if store_1 else "Store 1"
        name_2 = store_2.get("name", "Store 2") if store_2 else "Store 2"

        # Query each store for information about the topic
        query = f"Подробно опиши информацию о теме: {topic}. Приведи конкретные данные, цифры, требования из документов."

        result_1 = self.ask_question(store_id_1, query, model)
        result_2 = self.ask_question(store_id_2, query, model)

        if not result_1 and not result_2:
            return f"Информация по теме '{topic}' не найдена ни в одном из тендеров."

        # Build comparison prompt
        comparison_prompt = f"""Сравни информацию по теме "{topic}" из двух тендеров.

**{name_1}:**
{result_1 or "Информация не найдена"}

**{name_2}:**
{result_2 or "Информация не найдена"}

Задача:
1. Сначала кратко опиши что содержит каждый тендер по этой теме
2. Выдели РАЗЛИЧИЯ между тендерами (разные требования, объемы, подходы)
3. Выдели СХОДСТВА (общие требования, одинаковые подходы)
4. Сделай краткий вывод

Ответ структурируй с заголовками."""

        try:
            response = self.client.models.generate_content(
                model=model,
                contents=comparison_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=2000,
                    thinking_config=types.ThinkingConfig(
                        thinking_level="medium"
                    )
                )
            )

            if response and response.text:
                logger.info(f"Generated comparison for topic: {topic}")
                return response.text

            return None

        except Exception as e:
            logger.error(f"Compare failed: {e}")
            return None


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
