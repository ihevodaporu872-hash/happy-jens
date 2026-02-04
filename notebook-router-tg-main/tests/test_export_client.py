from export_client import ExportClient


def test_clean_markdown():
    client = ExportClient()
    text = "**bold** and `code`\n\n```\nblock\n```"
    cleaned = client._clean_markdown(text)
    assert "**" not in cleaned
    assert "`" not in cleaned


def test_generate_filename():
    client = ExportClient()
    name = client._generate_filename("Test Export", ".pdf")
    assert name.endswith(".pdf")
    assert "Test_Export" in name
