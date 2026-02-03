"""
Notebooks Manager - handles notebook storage and URL parsing

Uses urllib.parse for URL handling and stores notebooks in JSON file.
"""

import json
import logging
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


def extract_notebook_id(url: str) -> Optional[str]:
    """
    Extract notebook ID from NotebookLM URL.

    Example URL: https://notebooklm.google.com/notebook/8b1d2368-449b-4bfc-abfa-5465b3a27981
    Returns: 8b1d2368-449b-4bfc-abfa-5465b3a27981
    """
    try:
        parsed = urlparse(url)

        # Check if it's a valid NotebookLM URL
        if "notebooklm.google.com" not in parsed.netloc:
            return None

        # Extract ID from path: /notebook/{id}
        path_parts = parsed.path.strip("/").split("/")

        if len(path_parts) >= 2 and path_parts[0] == "notebook":
            notebook_id = path_parts[1]
            # Validate UUID-like format
            if len(notebook_id) >= 32:
                return notebook_id

        return None
    except Exception as e:
        logger.error(f"Error parsing URL: {e}")
        return None


def load_notebooks(file_path: Path) -> List[Dict]:
    """Load notebooks from JSON file."""
    if not file_path.exists():
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        logger.error(f"Error loading notebooks: {e}")
        return []


def save_notebooks(file_path: Path, notebooks: List[Dict]) -> bool:
    """Save notebooks to JSON file."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(notebooks, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Error saving notebooks: {e}")
        return False


def notebook_exists(file_path: Path, notebook_id: str) -> bool:
    """Check if notebook with given ID already exists."""
    notebooks = load_notebooks(file_path)
    return any(nb.get("id") == notebook_id for nb in notebooks)


def add_notebook(file_path: Path, url: str, name: str, description: str) -> tuple[bool, str]:
    """
    Add a new notebook to the storage.

    Returns: (success, message)
    """
    # Extract ID from URL
    notebook_id = extract_notebook_id(url)
    if not notebook_id:
        return False, "Invalid NotebookLM URL. Expected format: https://notebooklm.google.com/notebook/{id}"

    # Check for duplicates
    if notebook_exists(file_path, notebook_id):
        return False, f"Notebook with ID {notebook_id} already exists"

    # Load existing notebooks
    notebooks = load_notebooks(file_path)

    # Add new notebook
    notebook = {
        "id": notebook_id,
        "url": url,
        "name": name,
        "description": description
    }
    notebooks.append(notebook)

    # Save
    if save_notebooks(file_path, notebooks):
        logger.info(f"Added notebook: {name} (ID: {notebook_id})")
        return True, f"Notebook '{name}' added successfully!"
    else:
        return False, "Failed to save notebook"


def remove_notebook(file_path: Path, notebook_id: str) -> tuple[bool, str]:
    """
    Remove a notebook by ID.

    Returns: (success, message)
    """
    notebooks = load_notebooks(file_path)

    # Find and remove
    for i, nb in enumerate(notebooks):
        if nb.get("id") == notebook_id:
            removed = notebooks.pop(i)
            if save_notebooks(file_path, notebooks):
                return True, f"Notebook '{removed.get('name', 'Unknown')}' removed"
            else:
                return False, "Failed to save after removal"

    return False, f"Notebook with ID {notebook_id} not found"


def get_notebook_by_id(file_path: Path, notebook_id: str) -> Optional[Dict]:
    """Get notebook by ID."""
    notebooks = load_notebooks(file_path)
    for nb in notebooks:
        if nb.get("id") == notebook_id:
            return nb
    return None


def list_notebooks(file_path: Path) -> List[Dict]:
    """Get all notebooks."""
    return load_notebooks(file_path)
