"""
Smart Router for NotebookLM using OpenAI API

Analyzes user questions and selects the most relevant notebooks.
"""

import json
import logging
from typing import List, Dict, Optional
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger(__name__)


class NotebookRouter:
    """
    Routes questions to the most relevant NotebookLM notebooks using GPT-4o-mini.
    """

    def __init__(self, api_key: str, library_path: Path):
        """
        Initialize the router.

        Args:
            api_key: OpenAI API key
            library_path: Path to library.json with notebook metadata
        """
        self.client = OpenAI(api_key=api_key)
        self.library_path = library_path
        self.notebooks: List[Dict] = []
        self._load_library()

    def _load_library(self):
        """Load notebook library from file"""
        if not self.library_path.exists():
            logger.warning(f"Library not found: {self.library_path}")
            return

        try:
            with open(self.library_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Support both formats: list or {"notebooks": []}
                if isinstance(data, list):
                    self.notebooks = data
                else:
                    self.notebooks = data.get("notebooks", [])
                logger.info(f"Loaded {len(self.notebooks)} notebooks")
        except Exception as e:
            logger.error(f"Failed to load library: {e}")

    def reload_library(self):
        """Reload library from disk"""
        self._load_library()

    def get_notebooks_summary(self) -> str:
        """Get formatted summary of all notebooks for routing"""
        if not self.notebooks:
            return "No notebooks available."

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
        Route a question to the most relevant notebooks.

        Args:
            question: User's question
            max_notebooks: Maximum number of notebooks to return

        Returns:
            List of relevant notebook dicts with URLs
        """
        if not self.notebooks:
            logger.warning("No notebooks in library")
            return []

        if len(self.notebooks) == 1:
            return self.notebooks[:1]

        # Build prompt
        notebooks_summary = self.get_notebooks_summary()

        prompt = f"""You are a routing assistant. Your task is to select the most relevant notebooks for a user's question.

Available notebooks:
{notebooks_summary}

User's question: "{question}"

Instructions:
1. Analyze the question and determine which notebook(s) would best answer it
2. Select 1 to {max_notebooks} most relevant notebooks
3. Return ONLY a JSON array with notebook names, ordered by relevance
4. If no notebook seems relevant, return an empty array []

IMPORTANT - Handle user input variations:
- Transliteration: "майприорити" = "MYPRIORITY", "дубровка" = "Dubrovka", "гранель" = "Granel"
- Typos and misspellings: understand intent even with errors
- Partial names: "приорити", "сезар", "событие" should match full names
- Mixed languages: user may mix Russian and English
- Case insensitive: ignore uppercase/lowercase differences
- Numbers: "295", "304", "6.2" are project identifiers

Example response: ["React Documentation", "Frontend Best Practices"]

Your response (JSON array only):"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse response
            content = response.choices[0].message.content.strip()

            # Try to extract JSON array
            if content.startswith("["):
                selected_names = json.loads(content)
            else:
                # Try to find JSON in response
                import re
                match = re.search(r'\[.*?\]', content, re.DOTALL)
                if match:
                    selected_names = json.loads(match.group())
                else:
                    logger.warning(f"Could not parse router response: {content}")
                    return self.notebooks[:max_notebooks]

            # Map names to notebook dicts with fuzzy matching
            selected = []
            logger.info(f"GPT selected: {selected_names}")

            for name in selected_names:
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

            if selected:
                logger.info(f"Router matched: {[nb.get('name') for nb in selected]}")
                return selected[:max_notebooks]
            else:
                # Fallback: return first notebooks
                logger.warning("Router found no matches, using fallback")
                return self.notebooks[:max_notebooks]

        except Exception as e:
            logger.error(f"Router error: {e}")
            # Fallback: return first notebooks
            return self.notebooks[:max_notebooks]

    def route_with_reasoning(self, question: str, max_notebooks: int = 3) -> tuple[List[Dict], str]:
        """
        Route with explanation of why notebooks were selected.

        Returns:
            Tuple of (selected notebooks, reasoning text)
        """
        if not self.notebooks:
            return [], "No notebooks available."

        notebooks_summary = self.get_notebooks_summary()

        prompt = f"""You are a routing assistant. Analyze the question and select relevant notebooks.

Available notebooks:
{notebooks_summary}

User's question: "{question}"

IMPORTANT - Handle user input variations:
- Transliteration: "майприорити" = "MYPRIORITY", "дубровка" = "Dubrovka", "гранель" = "Granel"
- Typos and misspellings: understand intent even with errors
- Partial names: "приорити", "сезар", "событие" should match full names
- Mixed languages: user may mix Russian and English
- Numbers: "295", "304", "6.2" are project identifiers

Respond in this exact JSON format:
{{
    "selected": ["Notebook Name 1", "Notebook Name 2"],
    "reasoning": "Brief explanation of why these notebooks were selected"
}}

Select 1-{max_notebooks} notebooks. Your response:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )

            content = response.choices[0].message.content.strip()

            # Parse JSON response
            import re
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                selected_names = data.get("selected", [])
                reasoning = data.get("reasoning", "")
            else:
                return self.route(question, max_notebooks), "Could not get reasoning"

            # Map names to notebooks with fuzzy matching
            selected = []
            logger.info(f"GPT selected names: {selected_names}")

            for name in selected_names:
                name_lower = name.lower().strip()
                best_match = None
                best_score = 0

                for nb in self.notebooks:
                    nb_name = nb.get("name", "").lower()

                    # Exact match
                    if nb_name == name_lower:
                        best_match = nb
                        break

                    # Partial match - check if selected name is contained in notebook name or vice versa
                    if name_lower in nb_name or nb_name in name_lower:
                        score = len(name_lower) / max(len(nb_name), 1)
                        if score > best_score:
                            best_score = score
                            best_match = nb

                    # Word overlap matching
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
                elif not best_match:
                    logger.warning(f"Could not match: '{name}'")

            return selected[:max_notebooks], reasoning

        except Exception as e:
            logger.error(f"Router error: {e}")
            return self.notebooks[:max_notebooks], f"Routing error: {e}"

    def enhance_prompt(self, question: str, notebook: Dict) -> str:
        """
        Enhance user's question for better NotebookLM response.
        
        Adds context about the notebook, requests detailed answer,
        and removes ambiguities.
        
        Args:
            question: Original user question
            notebook: Selected notebook dict with name, description, topics
            
        Returns:
            Enhanced prompt string
        """
        notebook_name = notebook.get("name", "Unknown")
        notebook_desc = notebook.get("description", "")
        topics = ", ".join(notebook.get("topics", [])) or "general"
        
        prompt = f"""Improve this question for a knowledge base query. 
The question will be sent to a NotebookLM notebook containing: {notebook_desc or notebook_name}

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
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=256,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            enhanced = response.choices[0].message.content.strip()
            
            # Remove quotes if GPT wrapped the response
            if enhanced.startswith('"') and enhanced.endswith('"'):
                enhanced = enhanced[1:-1]
            if enhanced.startswith("'") and enhanced.endswith("'"):
                enhanced = enhanced[1:-1]
                
            logger.info(f"Enhanced prompt: {enhanced[:100]}...")
            return enhanced
            
        except Exception as e:
            logger.error(f"Prompt enhancement failed: {e}")
            # Return original question on error
            return question


def add_notebook_to_library(
    library_path: Path,
    url: str,
    name: str,
    topics: List[str] = None,
    description: str = ""
) -> bool:
    """
    Add a notebook to the library.

    Args:
        library_path: Path to library.json
        url: NotebookLM URL
        name: Display name
        topics: List of topic tags
        description: Brief description of notebook contents

    Returns:
        True if added successfully
    """
    try:
        # Load existing library
        if library_path.exists():
            with open(library_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"notebooks": []}

        # Check for duplicates
        for nb in data.get("notebooks", []):
            if nb.get("url") == url:
                logger.info(f"Notebook already exists: {name}")
                return False

        # Add new notebook
        notebook = {
            "name": name,
            "url": url,
            "topics": topics or [],
            "description": description
        }
        data["notebooks"].append(notebook)

        # Ensure directory exists
        library_path.parent.mkdir(parents=True, exist_ok=True)

        # Save
        with open(library_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Added notebook: {name}")
        return True

    except Exception as e:
        logger.error(f"Failed to add notebook: {e}")
        return False


def generate_descriptions(api_key: str, library_path: Path) -> int:
    """
    Generate descriptions for notebooks that don't have one.

    Uses GPT-4o-mini to create descriptions based on notebook names.

    Args:
        api_key: OpenAI API key
        library_path: Path to library.json

    Returns:
        Number of descriptions generated
    """
    if not library_path.exists():
        return 0

    try:
        with open(library_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        notebooks = data.get("notebooks", [])
        needs_description = [nb for nb in notebooks if not nb.get("description")]

        if not needs_description:
            logger.info("All notebooks already have descriptions")
            return 0

        client = OpenAI(api_key=api_key)

        # Build list of names
        names = [nb.get("name", "Unknown") for nb in needs_description]

        prompt = f"""Generate brief descriptions for these NotebookLM notebooks based on their names.
These descriptions will help a routing system select the right notebook for user questions.

Notebooks:
{chr(10).join(f'- {name}' for name in names)}

For each notebook, provide a 10-20 word description of what topics/content it likely contains.
Respond in JSON format:
{{
    "descriptions": {{
        "Notebook Name": "Brief description of contents and topics",
        ...
    }}
}}"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.choices[0].message.content.strip()

        # Parse JSON
        import re
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if not match:
            logger.error("Could not parse GPT response")
            return 0

        result = json.loads(match.group())
        descriptions = result.get("descriptions", {})

        # Update notebooks
        updated = 0
        for nb in notebooks:
            name = nb.get("name", "")
            if name in descriptions and not nb.get("description"):
                nb["description"] = descriptions[name]
                updated += 1
                logger.info(f"Generated description for: {name}")

        # Save
        with open(library_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return updated

    except Exception as e:
        logger.error(f"Failed to generate descriptions: {e}")
        return 0
