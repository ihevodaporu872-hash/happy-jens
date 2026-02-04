"""
Smart Router using Gemini 3 Flash

Routes questions to the most relevant knowledge stores.
100% Gemini 3-powered - no OpenAI dependency.
"""

import json
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class NotebookRouter:
    """
    Routes questions to the most relevant stores using Gemini 3 Flash.
    """

    def __init__(self, api_key: str, library_path: Path, model: str = "gemini-3-flash-preview"):
        """
        Initialize the router.

        Args:
            api_key: Gemini API key
            library_path: Path to stores.json with store metadata
            model: Gemini model to use for routing
        """
        self.client = genai.Client(api_key=api_key)
        self.library_path = library_path
        self.model = model
        self.notebooks: List[Dict] = []
        self._load_library()

    def _load_library(self):
        """Load notebook/store library from file"""
        if not self.library_path.exists():
            logger.warning(f"Library not found: {self.library_path}")
            return

        try:
            with open(self.library_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.notebooks = data
                else:
                    self.notebooks = data.get("notebooks", [])
                logger.info(f"Loaded {len(self.notebooks)} stores for routing")
        except Exception as e:
            logger.error(f"Failed to load library: {e}")

    def reload_library(self):
        """Reload library from disk"""
        self._load_library()

    def get_notebooks_summary(self) -> str:
        """Get formatted summary of all stores for routing"""
        if not self.notebooks:
            return "No stores available."

        lines = []
        for i, nb in enumerate(self.notebooks, 1):
            name = nb.get("name", "Unnamed")
            topics = ", ".join(nb.get("topics", [])) or "No topics"
            description = nb.get("description", "No description")

            lines.append(f"{i}. **{name}**")
            lines.append(f"   Topics: {topics}")
            lines.append(f"   Description: {description}")
            lines.append("")

        return "\n".join(lines)

    def route(self, question: str, max_notebooks: int = 3) -> List[Dict]:
        """
        Route a question to the most relevant stores.

        Args:
            question: User's question
            max_notebooks: Maximum number of stores to return

        Returns:
            List of relevant store dicts
        """
        if not self.notebooks:
            logger.warning("No stores in library")
            return []

        if len(self.notebooks) == 1:
            return self.notebooks[:1]

        notebooks_summary = self.get_notebooks_summary()

        prompt = f"""You are a routing assistant. Select the most relevant knowledge stores for the user's question.

Available stores:
{notebooks_summary}

User's question: "{question}"

Instructions:
1. Analyze the question and determine which store(s) would best answer it
2. Select 1 to {max_notebooks} most relevant stores
3. Return ONLY a JSON array with store names, ordered by relevance
4. If no store seems relevant, return an empty array []

IMPORTANT - Handle user input variations:
- Transliteration: "майприорити" = "MYPRIORITY", "дубровка" = "Dubrovka"
- Typos and misspellings: understand intent even with errors
- Partial names: match partial names to full store names
- Mixed languages: user may mix Russian and English
- Case insensitive: ignore uppercase/lowercase differences

Example response: ["Documentation", "FAQ"]

Your response (JSON array only):"""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=256,
                    thinking_config=types.ThinkingConfig(
                        thinking_level="low"
                    )
                )
            )

            content = response.text.strip()

            # Parse JSON array
            if content.startswith("["):
                selected_names = json.loads(content)
            else:
                import re
                match = re.search(r'\[.*?\]', content, re.DOTALL)
                if match:
                    selected_names = json.loads(match.group())
                else:
                    logger.warning(f"Could not parse router response: {content}")
                    return self.notebooks[:max_notebooks]

            # Map names to store dicts with fuzzy matching
            selected = self._match_names_to_stores(selected_names, max_notebooks)

            if selected:
                logger.info(f"Router matched: {[nb.get('name') for nb in selected]}")
                return selected
            else:
                logger.warning("Router found no matches, using fallback")
                return self.notebooks[:max_notebooks]

        except Exception as e:
            logger.error(f"Router error: {e}")
            return self.notebooks[:max_notebooks]

    def route_with_reasoning(self, question: str, max_notebooks: int = 3) -> Tuple[List[Dict], str]:
        """
        Route with explanation of why stores were selected.

        Returns:
            Tuple of (selected stores, reasoning text)
        """
        if not self.notebooks:
            return [], "No stores available."

        if len(self.notebooks) == 1:
            return self.notebooks[:1], "Only one store available."

        notebooks_summary = self.get_notebooks_summary()

        prompt = f"""You are a routing assistant. Analyze the question and select relevant knowledge stores.

Available stores:
{notebooks_summary}

User's question: "{question}"

IMPORTANT - Handle user input variations:
- Transliteration: "майприорити" = "MYPRIORITY", "дубровка" = "Dubrovka"
- Typos and misspellings: understand intent even with errors
- Partial names: match partial names to full store names
- Mixed languages: user may mix Russian and English

Respond in this exact JSON format:
{{
    "selected": ["Store Name 1", "Store Name 2"],
    "reasoning": "Brief explanation of why these stores were selected"
}}

Select 1-{max_notebooks} stores. Your response:"""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=512,
                    thinking_config=types.ThinkingConfig(
                        thinking_level="low"
                    )
                )
            )

            content = response.text.strip()

            # Parse JSON response
            import re
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                selected_names = data.get("selected", [])
                reasoning = data.get("reasoning", "")
            else:
                return self.route(question, max_notebooks), "Could not get reasoning"

            selected = self._match_names_to_stores(selected_names, max_notebooks)
            return selected, reasoning

        except Exception as e:
            logger.error(f"Router error: {e}")
            return self.notebooks[:max_notebooks], f"Routing error: {e}"

    def _match_names_to_stores(self, names: List[str], max_results: int) -> List[Dict]:
        """Match selected names to actual store dicts with fuzzy matching."""
        selected = []

        for name in names:
            name_lower = name.lower().strip()
            best_match = None
            best_score = 0

            for nb in self.notebooks:
                nb_name = nb.get("name", "").lower()

                # Exact match
                if nb_name == name_lower:
                    best_match = nb
                    break

                # Partial match
                if name_lower in nb_name or nb_name in name_lower:
                    score = len(name_lower) / max(len(nb_name), 1)
                    if score > best_score:
                        best_score = score
                        best_match = nb

                # Word overlap
                name_words = set(name_lower.split())
                nb_words = set(nb_name.split())
                common_words = name_words & nb_words
                if common_words:
                    score = len(common_words) / max(len(name_words), len(nb_words))
                    if score > best_score:
                        best_score = score
                        best_match = nb

            if best_match and best_match not in selected:
                selected.append(best_match)
                logger.info(f"Matched '{name}' -> '{best_match.get('name')}'")

        return selected[:max_results]

    def enhance_prompt(self, question: str, notebook: Dict) -> str:
        """
        Enhance user's question for better response quality.

        Args:
            question: Original user question
            notebook: Selected store dict with name, description, topics

        Returns:
            Enhanced prompt string
        """
        notebook_name = notebook.get("name", "Unknown")
        notebook_desc = notebook.get("description", "")

        prompt = f"""Improve this question for a knowledge base query.
The question will be sent to a knowledge store containing: {notebook_desc or notebook_name}

Original question: "{question}"

Instructions:
1. Keep the core intent of the question
2. Make it more specific and clear
3. Add context if the question is too vague
4. Request a detailed, structured answer
5. If it's a simple factual question, keep it simple
6. Response MUST be in the same language as the original question
7. Return ONLY the improved question, nothing else

Improved question:"""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=256,
                    thinking_config=types.ThinkingConfig(
                        thinking_level="low"
                    )
                )
            )

            enhanced = response.text.strip()

            # Remove quotes if wrapped
            if enhanced.startswith('"') and enhanced.endswith('"'):
                enhanced = enhanced[1:-1]
            if enhanced.startswith("'") and enhanced.endswith("'"):
                enhanced = enhanced[1:-1]

            logger.info(f"Enhanced prompt: {enhanced[:100]}...")
            return enhanced

        except Exception as e:
            logger.error(f"Prompt enhancement failed: {e}")
            return question


def add_notebook_to_library(
    library_path: Path,
    store_id: str,
    name: str,
    topics: List[str] = None,
    description: str = ""
) -> bool:
    """
    Add a store to the library.

    Args:
        library_path: Path to stores.json
        store_id: Gemini store ID
        name: Display name
        topics: List of topic tags
        description: Brief description of store contents

    Returns:
        True if added successfully
    """
    try:
        if library_path.exists():
            with open(library_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []

        # Check for duplicates
        if isinstance(data, list):
            for nb in data:
                if nb.get("id") == store_id:
                    logger.info(f"Store already exists: {name}")
                    return False

            data.append({
                "id": store_id,
                "name": name,
                "topics": topics or [],
                "description": description
            })
        else:
            notebooks = data.get("notebooks", [])
            for nb in notebooks:
                if nb.get("id") == store_id:
                    logger.info(f"Store already exists: {name}")
                    return False
            notebooks.append({
                "id": store_id,
                "name": name,
                "topics": topics or [],
                "description": description
            })

        library_path.parent.mkdir(parents=True, exist_ok=True)

        with open(library_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Added store: {name}")
        return True

    except Exception as e:
        logger.error(f"Failed to add store: {e}")
        return False
