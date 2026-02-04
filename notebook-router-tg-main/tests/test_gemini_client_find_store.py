from gemini_client import GeminiFileSearchClient


def test_find_store_by_name_fuzzy():
    client = GeminiFileSearchClient.__new__(GeminiFileSearchClient)
    client.stores = [
        {"id": "1", "name": "Tender Dubrovka"},
        {"id": "2", "name": "MyPriority"},
    ]

    assert client.find_store_by_name("Dubrovka")["id"] == "1"
    assert client.find_store_by_name("mypriority")["id"] == "2"
