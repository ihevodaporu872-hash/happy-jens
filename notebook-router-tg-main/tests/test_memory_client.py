from memory_client import UserMemoryClient


def test_memory_add_and_context(tmp_path):
    path = tmp_path / "memory.json"
    client = UserMemoryClient(path, max_messages=2)

    client.add_message(1, "store1", "user", "hello")
    client.add_message(1, "store1", "assistant", "hi")
    client.add_message(1, "store1", "user", "next")

    history = client.get_history(1, "store1")
    assert len(history) == 2
    assert history[0]["content"] == "hi"

    context = client.get_context_prompt(1, "store1")
    assert "Пользователь" in context
    assert "Ассистент" in context
