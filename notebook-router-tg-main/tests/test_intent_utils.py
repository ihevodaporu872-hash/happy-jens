from intent_utils import infer_action_from_text, extract_target_store_hint


def test_list_stores_action():
    action, args = infer_action_from_text("Покажи список тендеров")
    assert action == "list_stores"
    assert args == {}


def test_select_store_action():
    action, args = infer_action_from_text("Выбери тендер Дубровка")
    assert action == "select_store"
    assert args.get("store_name") == "Дубровка"


def test_rename_store_action():
    action, args = infer_action_from_text("Переименуй тендер Дубровка в Дубровка 2026")
    assert action == "rename_store"
    assert args.get("old_name") == "Дубровка"
    assert args.get("new_name") == "Дубровка 2026"


def test_delete_store_action():
    action, args = infer_action_from_text("Удалить тендер Тест")
    assert action == "delete_store"
    assert args.get("store_name") == "Тест"


def test_export_action_format():
    action, args = infer_action_from_text("Сделай экспорт в PDF")
    assert action == "export"
    assert args.get("format") == "pdf"


def test_extract_target_store_hint():
    store = extract_target_store_hint("В тендере Дубровка какие сроки?")
    assert store == "Дубровка"
