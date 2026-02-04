"""
Prompt Enhancer Module using Gemini 3 Flash

Enhances user prompts for better RAG responses.
Uses template matching first, then AI enhancement as fallback.
"""

import json
import logging
from typing import Optional, Tuple, List
from pathlib import Path

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class PromptEnhancer:
    """
    Enhances prompts using Gemini 3 Flash with low thinking.
    """

    def __init__(self, api_key: str, prompts_dir: Optional[Path] = None, model: str = "gemini-3-flash-preview"):
        """
        Initialize the enhancer.

        Args:
            api_key: Gemini API key
            prompts_dir: Optional directory with prompts_library.json
            model: Gemini model to use
        """
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.prompts_library = {}

        if prompts_dir:
            library_path = prompts_dir / "prompts_library.json"
            if library_path.exists():
                try:
                    with open(library_path, "r", encoding="utf-8") as f:
                        self.prompts_library = json.load(f)
                    logger.info(f"Loaded prompts library with {len(self.prompts_library.get('sections', {}))} sections")
                except Exception as e:
                    logger.error(f"Error loading prompts library: {e}")

    def find_matching_template(self, query: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Find a matching ideal prompt template for the user query.

        Returns:
            Tuple of (section_name, prompt_type, template) or (None, None, None) if no match
        """
        query_lower = query.lower()
        sections = self.prompts_library.get("sections", {})

        best_match = None
        best_score = 0

        for section_key, section_data in sections.items():
            section_keywords = section_data.get("keywords", [])
            section_score = sum(1 for kw in section_keywords if kw.lower() in query_lower)

            if section_score > 0:
                prompts = section_data.get("prompts", {})

                for prompt_key, prompt_data in prompts.items():
                    prompt_keywords = prompt_data.get("keywords", [])
                    prompt_score = sum(1 for kw in prompt_keywords if kw.lower() in query_lower)

                    total_score = section_score + prompt_score * 2

                    if total_score > best_score:
                        best_score = total_score
                        best_match = (section_key, prompt_data.get("name"), prompt_data.get("template"))

        if best_match and best_score >= 2:
            logger.info(f"Found template match: section='{best_match[0]}', type='{best_match[1]}', score={best_score}")
            return best_match

        return None, None, None

    def enhance(self, query: str, context_notebooks: List[str]) -> str:
        """
        Enhance the user query for better RAG response.

        First tries to find a matching ideal prompt template.
        If no match, uses Gemini to improve the query.

        Args:
            query: Original user query
            context_notebooks: List of store names selected by router

        Returns:
            Enhanced query string
        """
        if not query:
            return ""

        # Step 1: Try to find matching template
        section_name, prompt_type, template = self.find_matching_template(query)

        if template:
            logger.info(f"Using existing template for '{section_name}' - '{prompt_type}'")
            return template

        # Step 2: No matching template, use Gemini enhancement
        context_str = ", ".join(context_notebooks) if context_notebooks else "General Knowledge"

        prompt = f"""You are an expert prompt engineer for RAG systems.
Your goal is to rewrite the user's query to maximize quality and accuracy of answers.

Target Context: {context_str}

Guidelines:
1. Make the query SPECIFIC and DETAILED
2. Add relevant context that might be implied
3. Use keywords likely in source documents
4. Expand simple questions to ask for explanation, examples, evidence
5. Structure for comprehensive response
6. Language: Keep the same language as the original query

Original Query: "{query}"

Return ONLY the enhanced query text, nothing else."""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=800,
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

            logger.info(f"Gemini enhanced query: '{query[:50]}...' -> '{enhanced[:50]}...'")
            return enhanced

        except Exception as e:
            logger.error(f"Error enhancing prompt: {e}")
            return query
