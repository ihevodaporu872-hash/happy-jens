"""
Smart Query Processor with Ultrathinking

Uses Gemini with high thinking level to:
1. Understand user intent even with typos/errors
2. Detect query type (single store, multistore, web search, compare, sources)
3. Generate optimal prompt for the knowledge store
"""

import json
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


@dataclass
class ProcessedQuery:
    """Result of query processing"""
    query_type: str  # "single", "multistore", "web_search", "compare", "sources"
    optimized_prompt: str  # Ideal prompt for the store
    include_sources: bool  # Whether to include source citations
    target_stores: Optional[List[str]]  # For compare: store names
    compare_topic: Optional[str]  # For compare: topic
    original_question: str
    user_intent: str  # Brief description of what user wants
    confidence: float  # 0-1 confidence in interpretation


class QueryProcessor:
    """
    Smart query processor using Gemini ultrathinking.

    Understands user intent even with:
    - Typos and spelling errors
    - Transliteration (Russian/English mix)
    - Incomplete or vague questions
    - Implicit context from conversation
    """

    def __init__(self, api_key: str, model: str = "gemini-3-flash-preview"):
        """
        Initialize the processor.

        Args:
            api_key: Gemini API key
            model: Gemini model to use
        """
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def process_query(
        self,
        question: str,
        available_stores: List[Dict],
        conversation_context: str = ""
    ) -> ProcessedQuery:
        """
        Process user query with ultrathinking to understand intent and optimize prompt.

        Args:
            question: Raw user question (may contain errors)
            available_stores: List of available stores with names and descriptions
            conversation_context: Previous conversation for context

        Returns:
            ProcessedQuery with type, optimized prompt, and metadata
        """
        # Build stores info for the prompt
        stores_info = self._format_stores_info(available_stores)

        analysis_prompt = f"""Ты - эксперт по анализу запросов пользователей для системы поиска по тендерной документации.

ДОСТУПНЫЕ БАЗЫ ЗНАНИЙ (stores):
{stores_info}

КОНТЕКСТ ПРЕДЫДУЩЕГО ДИАЛОГА:
{conversation_context if conversation_context else "Нет предыдущего контекста"}

ЗАПРОС ПОЛЬЗОВАТЕЛЯ (может содержать ошибки, опечатки, транслит):
"{question}"

ТВОЯ ЗАДАЧА:
1. ПОНЯТЬ что именно хочет пользователь, даже если он написал с ошибками
2. ОПРЕДЕЛИТЬ тип запроса
3. СФОРМИРОВАТЬ идеальный промпт для получения лучшего ответа

ТИПЫ ЗАПРОСОВ:
- "single" - вопрос к одному конкретному store (определи какому)
- "multistore" - поиск информации по ВСЕМ stores (например: "где есть...", "в каких тендерах...", "найди везде...")
- "web_search" - нужен поиск в интернете (актуальные цены, новости, текущие нормативы)
- "compare" - сравнение двух stores по теме
- "sources" - пользователь просит указать источники/ссылки

ВАЖНО:
- Пользователь может писать "майприорити" вместо "MYPRIORITY"
- Может писать "водопанижение" вместо "водопонижение"
- Может использовать сокращения и сленг
- Может задавать неполные вопросы опираясь на контекст

Ответь СТРОГО в формате JSON:
{{
    "query_type": "single|multistore|web_search|compare|sources",
    "user_intent": "краткое описание что хочет пользователь",
    "optimized_prompt": "идеальный развёрнутый промпт для получения полного ответа из базы знаний",
    "include_sources": true/false,
    "target_store": "имя store для single запроса или null",
    "compare_stores": ["store1", "store2"] или null,
    "compare_topic": "тема сравнения или null",
    "confidence": 0.0-1.0
}}

ПРАВИЛА ФОРМИРОВАНИЯ optimized_prompt:
1. Исправь все ошибки и опечатки
2. Разверни вопрос, добавь контекст
3. Попроси конкретные данные: цифры, сроки, объёмы
4. Укажи что нужен структурированный ответ
5. Сохрани язык пользователя (русский)

JSON:"""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=analysis_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=1500,
                    thinking_config=types.ThinkingConfig(
                        thinking_level="high"  # Ultrathinking for best understanding
                    )
                )
            )

            # Parse JSON response
            result = self._parse_response(response.text, question)
            logger.info(f"Query processed: type={result.query_type}, intent='{result.user_intent}'")
            return result

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            # Fallback to basic processing
            return ProcessedQuery(
                query_type="single",
                optimized_prompt=question,
                include_sources=False,
                target_stores=None,
                compare_topic=None,
                original_question=question,
                user_intent="Не удалось определить",
                confidence=0.0
            )

    def _format_stores_info(self, stores: List[Dict]) -> str:
        """Format stores list for the prompt."""
        if not stores:
            return "Нет доступных баз знаний"

        lines = []
        for i, store in enumerate(stores, 1):
            name = store.get("name", "Unknown")
            desc = store.get("description", "Без описания")
            docs_count = len(store.get("documents", []))
            lines.append(f"{i}. {name} - {desc} (документов: {docs_count})")

        return "\n".join(lines)

    def _parse_response(self, response_text: str, original_question: str) -> ProcessedQuery:
        """Parse JSON response from Gemini."""
        import re

        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if not json_match:
            raise ValueError("No JSON found in response")

        data = json.loads(json_match.group())

        return ProcessedQuery(
            query_type=data.get("query_type", "single"),
            optimized_prompt=data.get("optimized_prompt", original_question),
            include_sources=data.get("include_sources", False),
            target_stores=data.get("compare_stores"),
            compare_topic=data.get("compare_topic"),
            original_question=original_question,
            user_intent=data.get("user_intent", ""),
            confidence=float(data.get("confidence", 0.5))
        )

    def enhance_for_store(
        self,
        question: str,
        store_name: str,
        store_description: str
    ) -> str:
        """
        Enhance a question specifically for a store using ultrathinking.

        Args:
            question: User's question
            store_name: Target store name
            store_description: Store description

        Returns:
            Enhanced prompt optimized for the store
        """
        enhance_prompt = f"""Ты эксперт по тендерной документации. Улучши вопрос пользователя для поиска в базе знаний.

БАЗА ЗНАНИЙ: {store_name}
ОПИСАНИЕ: {store_description}

ВОПРОС ПОЛЬЗОВАТЕЛЯ (может содержать ошибки):
"{question}"

ЗАДАЧА:
1. Исправь ошибки и опечатки
2. Разверни вопрос для получения полного ответа
3. Добавь запрос конкретных данных (цифры, сроки, требования)
4. Попроси структурированный ответ с разделами

Верни ТОЛЬКО улучшенный вопрос, без пояснений:"""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=enhance_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=500,
                    thinking_config=types.ThinkingConfig(
                        thinking_level="high"
                    )
                )
            )

            enhanced = response.text.strip()
            # Remove quotes if wrapped
            if enhanced.startswith('"') and enhanced.endswith('"'):
                enhanced = enhanced[1:-1]

            logger.info(f"Enhanced question for {store_name}")
            return enhanced

        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            return question
