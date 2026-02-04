"""
Smart Query Processor

Uses Gemini 3 Pro for complex understanding:
1. Understand user intent even with typos/errors
2. Detect query type (single store, multistore, web search, compare, sources)
3. Generate optimal prompt for the knowledge store

Model strategy:
- Gemini 3 Pro (gemini-3-pro-preview): Query analysis, intent detection, prompt optimization
- Gemini 3 Flash (gemini-3-flash-preview): Simple/medium store queries (handled elsewhere)
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
    target_store: Optional[str]  # For single queries: store name
    target_stores: Optional[List[str]]  # For compare: store names
    compare_topic: Optional[str]  # For compare: topic
    original_question: str
    user_intent: str  # Brief description of what user wants
    confidence: float  # 0-1 confidence in interpretation
    complexity: str  # "simple", "medium", "complex" - for model selection
    action: Optional[str]  # Command-like action intent
    action_args: Dict  # Args for action


class QueryProcessor:
    """
    Smart query processor using Gemini Pro model.

    Understands user intent even with:
    - Typos and spelling errors
    - Transliteration (Russian/English mix)
    - Incomplete or vague questions
    - Implicit context from conversation
    """

    def __init__(self, api_key: str, model: str = "gemini-3-pro-preview"):
        """
        Initialize the processor.

        Args:
            api_key: Gemini API key
            model: Gemini 3 Pro model for complex analysis
        """
        self.client = genai.Client(api_key=api_key)
        self.model = model  # Gemini 3 Pro for understanding

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
    "action": "none|list_stores|select_store|status|clear_memory|export|add_store|delete_store|rename_store|set_sync|sync_now|upload_url|upload_file|help",
    "action_args": {{ "store_name": "...", "old_name": "...", "new_name": "...", "urls": ["..."], "format": "pdf|docx", "question": "..." }},
    "confidence": 0.0-1.0,
    "complexity": "simple|medium|complex"
}}

ПРАВИЛО ДЛЯ action:
- Используй action ТОЛЬКО если пользователь явно просит выполнить действие (список, выбор, экспорт, переименование и т.п.)
- Если это обычный вопрос к документам — action = "none"

ПРАВИЛА ФОРМИРОВАНИЯ optimized_prompt:
1. Исправь все ошибки и опечатки
2. Разверни вопрос, добавь контекст
3. Попроси конкретные данные: цифры, сроки, объёмы
4. Укажи что нужен структурированный ответ
5. Сохрани язык пользователя (русский)

ОПРЕДЕЛИ СЛОЖНОСТЬ ЗАПРОСА:
- "simple" - простой факт, одно значение (Какой срок? Какая цена?)
- "medium" - требует поиска и структурирования (Опиши требования к...)
- "complex" - сравнение, анализ, синтез из нескольких источников

JSON:"""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=analysis_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=1500
                )
            )

            # Parse JSON response
            result = self._parse_response(response.text, question)
            logger.info(f"Query processed: type={result.query_type}, complexity={result.complexity}, "
                       f"intent='{result.user_intent}'")
            return result

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            # Fallback to basic processing
            return ProcessedQuery(
                query_type="single",
                optimized_prompt=question,
                include_sources=False,
                target_store=None,
                target_stores=None,
                compare_topic=None,
                original_question=question,
                user_intent="Не удалось определить",
                confidence=0.0,
                complexity="medium",
                action=None,
                action_args={}
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
            target_store=data.get("target_store"),
            target_stores=data.get("compare_stores"),
            compare_topic=data.get("compare_topic"),
            original_question=original_question,
            user_intent=data.get("user_intent", ""),
            confidence=float(data.get("confidence", 0.5)),
            complexity=data.get("complexity", "medium"),
            action=data.get("action"),
            action_args=data.get("action_args") or {}
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
                    max_output_tokens=500
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
