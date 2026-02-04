import re
from typing import Dict, Optional, Tuple

ActionResult = Tuple[Optional[str], Dict[str, str]]


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clean_store_name(name: str) -> str:
    name = name.strip().strip('"\'`')
    # Remove trailing polite words/punctuation
    name = re.sub(r"[\s,.!?;:]+$", "", name)
    return name.strip()


def infer_action_from_text(text: str) -> ActionResult:
    """
    Heuristic fallback to infer command-like actions from natural language.
    Returns (action, action_args) or (None, {}).
    """
    if not text:
        return None, {}

    original = _clean_text(text)
    lower = original.lower()

    # List stores
    if re.search(
        r"\b(список|перечисли|покажи|какие|какие есть)\b.*\b(тендер|тендеры|тендеров|тендерах|store|stores|баз|проект)\b",
        lower
    ):
        return "list_stores", {}

    # Status
    if re.search(r"\b(статус|провер(ь|ка)|состояние)\b", lower):
        return "status", {}

    # Clear memory
    if re.search(r"\b(очист(и|ить)|сброс(ь|ить))\b.*\b(истор|памят|контекст)\b", lower):
        return "clear_memory", {}

    # Export
    if re.search(r"\b(экспорт|выгруз|сохрани|сохранить|export|сделай файл|сделай pdf|сделай docx)\b", lower):
        fmt = None
        if re.search(r"\b(pdf|пдф)\b", lower):
            fmt = "pdf"
        elif re.search(r"\b(docx|докх|док)\b", lower):
            fmt = "docx"
        return "export", {"format": fmt} if fmt else {}

    # Select store
    m = re.search(r"\b(выбери|выбрать|используй|переключи(?:сь)?|работай с|сделай активным|установи)\b\s*(?:тендер|store|баз[ау])?\s*(.+)", original, re.IGNORECASE)
    if m:
        store_name = _clean_store_name(m.group(2))
        if store_name:
            return "select_store", {"store_name": store_name}

    # Rename store
    m = re.search(
        r"\b(переименуй|переименовать|rename)\b\s*(?:тендер|store|баз[ау])?\s*(.+?)\s+(?:в|на)\s+(.+)",
        original,
        re.IGNORECASE
    )
    if m:
        old_name = _clean_store_name(m.group(2))
        new_name = _clean_store_name(m.group(3))
        if old_name and new_name:
            return "rename_store", {"old_name": old_name, "new_name": new_name}

    # Delete store
    m = re.search(r"\b(удали|удалить|delete|снеси)\b\s*(?:тендер|store|баз[ау])?\s*(.+)", original, re.IGNORECASE)
    if m:
        store_name = _clean_store_name(m.group(2))
        if store_name:
            return "delete_store", {"store_name": store_name}

    return None, {}


def extract_target_store_hint(text: str) -> Optional[str]:
    """Try to extract a store name hint from natural language."""
    if not text:
        return None

    # Patterns like: "в тендере X", "по тендеру X", "для тендера X"
    patterns = [
        r"\bв\s+тендер[еа]?\s+([^\n,.!?]+)",
        r"\bпо\s+тендер[уе]?\s+([^\n,.!?]+)",
        r"\bдля\s+тендер[а]?\s+([^\n,.!?]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = _clean_store_name(match.group(1))
            # Trim common question continuations
            name = re.sub(
                r"\b(что|какие|какой|когда|сколько|нужно|есть|требования|сроки|цены|стоимость)\b.*$",
                "",
                name,
                flags=re.IGNORECASE
            ).strip()
            return _clean_store_name(name)

    return None
