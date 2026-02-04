"""
Google Drive Client for downloading files by URL.

Supports:
- Google Docs -> PDF
- Google Sheets -> XLSX
- Google Slides -> PDF
- Regular files from Drive
- Folders (recursive download)

Works with:
- Service Account credentials (full API access)
- Public URLs (no auth required for "Anyone with the link" files)
"""

import io
import logging
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import requests
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

logger = logging.getLogger(__name__)

# Google MIME types and export formats
GOOGLE_MIME_TYPES = {
    'application/vnd.google-apps.document': {
        'export_mime': 'application/pdf',
        'extension': '.pdf'
    },
    'application/vnd.google-apps.spreadsheet': {
        'export_mime': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'extension': '.xlsx'
    },
    'application/vnd.google-apps.presentation': {
        'export_mime': 'application/pdf',
        'extension': '.pdf'
    },
    'application/vnd.google-apps.drawing': {
        'export_mime': 'application/pdf',
        'extension': '.pdf'
    }
}

# URL patterns for Google services
URL_PATTERNS = [
    # Google Docs
    (r'docs\.google\.com/document/d/([a-zA-Z0-9_-]+)', 'document'),
    # Google Sheets
    (r'docs\.google\.com/spreadsheets/d/([a-zA-Z0-9_-]+)', 'spreadsheet'),
    # Google Slides
    (r'docs\.google\.com/presentation/d/([a-zA-Z0-9_-]+)', 'presentation'),
    # Google Drive file
    (r'drive\.google\.com/file/d/([a-zA-Z0-9_-]+)', 'file'),
    # Google Drive open
    (r'drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)', 'file'),
    # Google Drive folder
    (r'drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)', 'folder'),
    # Google Drive folder (u/0 variant)
    (r'drive\.google\.com/drive/u/\d+/folders/([a-zA-Z0-9_-]+)', 'folder'),
]

# Limits
MAX_URLS_PER_REQUEST = 10
MAX_FILES_PER_FOLDER = 50

# Public export URLs (no auth required for "Anyone with the link" files)
PUBLIC_EXPORT_URLS = {
    'document': 'https://docs.google.com/document/d/{id}/export?format=pdf',
    'spreadsheet': 'https://docs.google.com/spreadsheets/d/{id}/export?format=xlsx',
    'presentation': 'https://docs.google.com/presentation/d/{id}/export/pdf',
    'file': 'https://drive.google.com/uc?export=download&id={id}',
}

# Extensions for public downloads
PUBLIC_EXPORT_EXTENSIONS = {
    'document': '.pdf',
    'spreadsheet': '.xlsx',
    'presentation': '.pdf',
    'file': '',  # Keep original or detect from response
}


class GoogleDriveClient:
    """
    Client for downloading files from Google Drive using Service Account.
    """

    def __init__(self, credentials_file: str):
        """
        Initialize the client.

        Args:
            credentials_file: Path to service account JSON file
        """
        self.credentials_file = Path(credentials_file)
        self.service = None
        self._init_service()

    def _init_service(self):
        """Initialize Google Drive API service."""
        if not self.credentials_file.exists():
            logger.warning(f"Service account file not found: {self.credentials_file}")
            return

        try:
            credentials = service_account.Credentials.from_service_account_file(
                str(self.credentials_file),
                scopes=['https://www.googleapis.com/auth/drive.readonly']
            )
            self.service = build('drive', 'v3', credentials=credentials)
            logger.info("Google Drive API initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Google Drive API: {e}")
            self.service = None

    def is_configured(self) -> bool:
        """Check if the client is properly configured with Service Account."""
        return self.service is not None

    @staticmethod
    def download_public_file(file_id: str, file_type: str, dest_dir: Path) -> Optional[Path]:
        """
        Download file via public export URL (no auth required).
        Works only for files with "Anyone with the link can view" access.

        Args:
            file_id: Google Drive file ID
            file_type: Type of file ('document', 'spreadsheet', 'presentation', 'file')
            dest_dir: Directory to save the file

        Returns:
            Path to downloaded file or None on failure
        """
        if file_type not in PUBLIC_EXPORT_URLS:
            logger.warning(f"Unsupported file type for public download: {file_type}")
            return None

        url = PUBLIC_EXPORT_URLS[file_type].format(id=file_id)

        try:
            response = requests.get(url, timeout=60, allow_redirects=True)

            if response.status_code != 200:
                logger.warning(f"Public download failed for {file_id}: HTTP {response.status_code}")
                return None

            # Check if we got an HTML error page instead of the file
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' in content_type and file_type != 'file':
                logger.warning(f"Public download failed for {file_id}: got HTML instead of file (access denied?)")
                return None

            # Extract filename from Content-Disposition header
            filename = None
            content_disp = response.headers.get('Content-Disposition', '')
            if content_disp:
                # Try to extract filename from header
                match = re.search(r"filename\*?=['\"]?(?:UTF-8'')?([^'\";\n]+)", content_disp)
                if match:
                    filename = match.group(1)
                    # URL decode if needed
                    try:
                        from urllib.parse import unquote
                        filename = unquote(filename)
                    except Exception:
                        pass

            # Generate filename if not found
            if not filename:
                extension = PUBLIC_EXPORT_EXTENSIONS.get(file_type, '')
                filename = f"{file_id}{extension}"

            # Sanitize filename
            filename = GoogleDriveClient._sanitize_filename(filename)

            # Save file
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / filename
            dest_path.write_bytes(response.content)

            logger.info(f"Downloaded (public): {filename} -> {dest_path}")
            return dest_path

        except requests.RequestException as e:
            logger.error(f"Public download error for {file_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in public download for {file_id}: {e}")
            return None

    @staticmethod
    def extract_file_id(url: str) -> Optional[Tuple[str, str]]:
        """
        Extract file ID and type from a Google URL.

        Args:
            url: Google Docs/Sheets/Drive URL

        Returns:
            Tuple of (file_id, type) or None if not a valid Google URL
        """
        for pattern, file_type in URL_PATTERNS:
            match = re.search(pattern, url)
            if match:
                return match.group(1), file_type
        return None

    @staticmethod
    def extract_all_urls(text: str) -> List[Tuple[str, str, str]]:
        """
        Extract all Google URLs from text.

        Args:
            text: Text containing URLs

        Returns:
            List of tuples (url, file_id, type)
        """
        results = []
        seen_ids = set()

        # Find all URLs in text (including query params like ?usp=sharing)
        url_pattern = r'https?://[^\s<>"\']+(?<=[a-zA-Z0-9_/=-])'
        urls = re.findall(url_pattern, text)
        # Clean trailing punctuation that might be captured
        urls = [re.sub(r'[,;:!?\)]+$', '', url) for url in urls]

        logger.debug(f"Found raw URLs: {urls}")

        for url in urls:
            extracted = GoogleDriveClient.extract_file_id(url)
            logger.debug(f"URL: {url[:50]}... -> extracted: {extracted}")
            if extracted:
                file_id, file_type = extracted
                if file_id not in seen_ids:
                    seen_ids.add(file_id)
                    results.append((url, file_id, file_type))

        logger.debug(f"Final extracted: {[(fid[:10], ftype) for _, fid, ftype in results]}")
        return results[:MAX_URLS_PER_REQUEST]

    def get_file_info(self, file_id: str) -> Optional[Dict]:
        """
        Get file metadata from Google Drive.

        Args:
            file_id: Google Drive file ID

        Returns:
            File metadata dict or None
        """
        if not self.service:
            logger.error("Google Drive not configured")
            return None

        try:
            file_info = self.service.files().get(
                fileId=file_id,
                fields='id,name,mimeType,size,modifiedTime'
            ).execute()
            return file_info
        except Exception as e:
            logger.error(f"Failed to get file info for {file_id}: {e}")
            return None

    def download_file(self, file_id: str, dest_dir: Path, file_type: Optional[str] = None) -> Optional[Path]:
        """
        Download a file from Google Drive.

        For Google Docs/Sheets/Slides, exports to PDF/XLSX.
        For regular files, downloads as-is.

        If Service Account is not configured, falls back to public URL download
        (works only for files with "Anyone with the link" access).

        Args:
            file_id: Google Drive file ID
            dest_dir: Directory to save the file
            file_type: Optional file type hint for public download fallback

        Returns:
            Path to downloaded file or None on failure
        """
        # Fallback to public download if Service Account not configured
        if not self.service:
            if file_type:
                logger.info(f"No Service Account, trying public download for {file_id}")
                return self.download_public_file(file_id, file_type, dest_dir)
            else:
                logger.error("Google Drive not configured and no file_type provided for public fallback")
                return None

        try:
            # Get file metadata
            file_info = self.get_file_info(file_id)
            if not file_info:
                return None

            mime_type = file_info.get('mimeType', '')
            file_name = file_info.get('name', file_id)

            # Check if it's a Google native format
            if mime_type in GOOGLE_MIME_TYPES:
                export_config = GOOGLE_MIME_TYPES[mime_type]
                export_mime = export_config['export_mime']
                extension = export_config['extension']

                # Add extension if not present
                if not file_name.lower().endswith(extension):
                    file_name = file_name + extension

                # Export the file
                request = self.service.files().export_media(
                    fileId=file_id,
                    mimeType=export_mime
                )
            else:
                # Regular file - download as-is
                request = self.service.files().get_media(fileId=file_id)

            # Download to memory
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / self._sanitize_filename(file_name)

            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()

            # Write to file
            with open(dest_path, 'wb') as f:
                f.write(fh.getvalue())

            logger.info(f"Downloaded: {file_name} -> {dest_path}")
            return dest_path

        except Exception as e:
            logger.error(f"Failed to download file {file_id}: {e}")
            return None

    def list_folder(self, folder_id: str, recursive: bool = True) -> List[Dict]:
        """
        List all files in a Google Drive folder.

        Args:
            folder_id: Folder ID
            recursive: Include files from subfolders

        Returns:
            List of file metadata dicts
        """
        if not self.service:
            logger.error("Google Drive not configured")
            return []

        files = []
        folders_to_process = [folder_id]

        while folders_to_process and len(files) < MAX_FILES_PER_FOLDER:
            current_folder = folders_to_process.pop(0)

            try:
                page_token = None
                while True:
                    response = self.service.files().list(
                        q=f"'{current_folder}' in parents and trashed = false",
                        fields='nextPageToken, files(id, name, mimeType, size)',
                        pageSize=100,
                        pageToken=page_token
                    ).execute()

                    for item in response.get('files', []):
                        if item['mimeType'] == 'application/vnd.google-apps.folder':
                            if recursive:
                                folders_to_process.append(item['id'])
                        else:
                            files.append(item)
                            if len(files) >= MAX_FILES_PER_FOLDER:
                                break

                    page_token = response.get('nextPageToken')
                    if not page_token or len(files) >= MAX_FILES_PER_FOLDER:
                        break

            except Exception as e:
                logger.error(f"Failed to list folder {current_folder}: {e}")

        return files

    def download_folder(self, folder_id: str, dest_dir: Path) -> List[Tuple[Path, str]]:
        """
        Download all files from a Google Drive folder.

        Args:
            folder_id: Folder ID
            dest_dir: Directory to save files

        Returns:
            List of tuples (file_path, file_name) for successful downloads
        """
        files = self.list_folder(folder_id, recursive=True)
        downloaded = []

        for file_info in files:
            file_path = self.download_file(file_info['id'], dest_dir)
            if file_path:
                downloaded.append((file_path, file_info['name']))

        return downloaded

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Sanitize filename for filesystem."""
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        return filename[:200]  # Limit length


# CLI for testing
if __name__ == "__main__":
    import sys
    import os
    from dotenv import load_dotenv

    load_dotenv()

    credentials_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "service_account.json")
    client = GoogleDriveClient(credentials_file)

    if not client.is_configured():
        print("Error: Google Drive not configured")
        print(f"Make sure {credentials_file} exists")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Google Drive Client CLI")
        print()
        print("Usage:")
        print("  python google_drive_client.py info <url>")
        print("  python google_drive_client.py download <url> [dest_dir]")
        print("  python google_drive_client.py folder <folder_url> [dest_dir]")
        print("  python google_drive_client.py extract <text_with_urls>")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "info" and len(sys.argv) >= 3:
        url = sys.argv[2]
        extracted = GoogleDriveClient.extract_file_id(url)
        if not extracted:
            print("Not a valid Google URL")
            sys.exit(1)
        file_id, file_type = extracted
        print(f"File ID: {file_id}")
        print(f"Type: {file_type}")

        info = client.get_file_info(file_id)
        if info:
            print(f"Name: {info.get('name')}")
            print(f"MIME: {info.get('mimeType')}")
            print(f"Size: {info.get('size', 'N/A')}")

    elif cmd == "download" and len(sys.argv) >= 3:
        url = sys.argv[2]
        dest_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("./downloads")

        extracted = GoogleDriveClient.extract_file_id(url)
        if not extracted:
            print("Not a valid Google URL")
            sys.exit(1)

        file_id, file_type = extracted
        if file_type == 'folder':
            print("Use 'folder' command for folders")
            sys.exit(1)

        print(f"Downloading {file_id}...")
        result = client.download_file(file_id, dest_dir)
        if result:
            print(f"Downloaded: {result}")
        else:
            print("Download failed")

    elif cmd == "folder" and len(sys.argv) >= 3:
        url = sys.argv[2]
        dest_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("./downloads")

        extracted = GoogleDriveClient.extract_file_id(url)
        if not extracted:
            print("Not a valid Google URL")
            sys.exit(1)

        file_id, file_type = extracted
        if file_type != 'folder':
            print("URL is not a folder")
            sys.exit(1)

        print(f"Listing folder {file_id}...")
        files = client.list_folder(file_id)
        print(f"Found {len(files)} files")

        print("Downloading...")
        downloaded = client.download_folder(file_id, dest_dir)
        print(f"Downloaded {len(downloaded)} files")
        for path, name in downloaded:
            print(f"  - {name}")

    elif cmd == "extract" and len(sys.argv) >= 3:
        text = " ".join(sys.argv[2:])
        urls = GoogleDriveClient.extract_all_urls(text)
        if not urls:
            print("No Google URLs found")
        else:
            print(f"Found {len(urls)} URLs:")
            for url, file_id, file_type in urls:
                print(f"  - {file_type}: {file_id}")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
