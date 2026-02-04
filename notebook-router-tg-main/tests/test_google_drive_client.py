from google_drive_client import GoogleDriveClient


def test_extract_file_id_document():
    url = "https://docs.google.com/document/d/abc123/edit"
    file_id, file_type = GoogleDriveClient.extract_file_id(url)
    assert file_id == "abc123"
    assert file_type == "document"


def test_extract_all_urls_limit():
    text = "Check https://docs.google.com/document/d/abc/edit and https://drive.google.com/file/d/xyz/view"
    urls = GoogleDriveClient.extract_all_urls(text)
    assert len(urls) == 2
