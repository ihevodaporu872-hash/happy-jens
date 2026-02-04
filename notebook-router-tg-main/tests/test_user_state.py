from user_state import UserStateClient


def test_user_state_set_get_clear(tmp_path):
    path = tmp_path / "state.json"
    client = UserStateClient(path)

    assert client.get_selected_store(1) is None

    client.set_selected_store(1, "store-1", "Tender A")
    data = client.get_selected_store(1)
    assert data["selected_store_id"] == "store-1"
    assert data["selected_store_name"] == "Tender A"

    client.clear_selected_store(1)
    assert client.get_selected_store(1) is None


def test_user_state_clear_store_for_all(tmp_path):
    path = tmp_path / "state.json"
    client = UserStateClient(path)

    client.set_selected_store(1, "store-1", "Tender A")
    client.set_selected_store(2, "store-2", "Tender B")
    client.clear_store_for_all("store-1")

    assert client.get_selected_store(1) is None
    assert client.get_selected_store(2)["selected_store_id"] == "store-2"
