# Scan File system
"""File system scanner"""
from pathlib import Path
from typing import List, Tuple
from backend.utils.file_utils import FileUtils
from backend.config.settings import settings
from backend.utils.logger import app_logger as logger


class FileScanner:
    """Scan and categorize files in directories"""

    def __init__(self):
        self.file_utils = FileUtils()

    def scan_folder(self, folder_path: str, recursive: bool = True) -> Tuple[List[str], List[str], List[str]]:
        """Scan folder and categorize files"""
        logger.info(f"Scanning folder: {folder_path}")

        all_files: List[str] = []
        text_files: List[str] = []
        image_files: List[str] = []

        path = Path(folder_path)

        if not path.exists() or not path.is_dir():
            logger.error(f"Folder does not exist or is not a directory: {folder_path}")
            return all_files, text_files, image_files

        excluded_exts = getattr(settings, "excluded_extensions", [])
        # max_size_mb = getattr(settings, "max_file_size_mb", None)

        file_iter = path.rglob("*") if recursive else path.glob("*")

        for file_path in file_iter:
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() in excluded_exts:
                continue

            try:
                stat_result = file_path.stat()
            except OSError as exc:
                logger.warning(f"Unable to stat file, skipping: {file_path} ({exc})")
                continue

            # if max_size_mb is not None and stat_result.st_size > max_size_mb * 1024 * 1024:
            #     logger.warning(f"File too large, skipping: {file_path}")
            #     continue

            try:
                file_type = self.file_utils.get_file_type(file_path)
            except Exception as exc:
                logger.warning(f"Unable to detect file type, skipping: {file_path} ({exc})")
                continue

            file_str = str(file_path)
            all_files.append(file_str)

            if file_type == "text":
                text_files.append(file_str)
            elif file_type == "image":
                image_files.append(file_str)

        logger.info(f"Found {len(all_files)} files: {len(text_files)} text, {len(image_files)} images")
        return all_files, text_files, image_files

    def scan_multiple_folders(self, folder_paths: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """Scan multiple folders"""
        all_files: List[str] = []
        text_files: List[str] = []
        image_files: List[str] = []

        for folder in folder_paths:
            a, t, i = self.scan_folder(folder)
            all_files.extend(a)
            text_files.extend(t)
            image_files.extend(i)

        return all_files, text_files, image_files
