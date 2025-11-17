"""
File utility functions
"""
from pathlib import Path
from typing import List, Optional
import hashlib
import mimetypes
from backend.utils.logger import app_logger as logger

class FileUtils:
    """File operation utilities"""
    
    SUPPORTED_TEXT_EXTENSIONS = {
        '.txt', '.md', '.pdf', '.doc', '.docx',
        '.ppt', '.pptx', '.xlsx', '.xls', '.csv'
    }
    
    SUPPORTED_IMAGE_EXTENSIONS = {
        '.jpg', '.jpeg', '.png', '.gif', '.bmp',
        '.tiff', '.webp', '.svg'
    }
    
    @staticmethod
    def get_file_hash(file_path: Path, algorithm: str = 'sha256') -> str:
        """Calculate file hash"""
        hash_func = hashlib.new(algorithm)
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hash_func.update(chunk)
            return hash_func.hexdigest()
        except Exception as e:
            logger.error(f"Error hashing file {file_path}: {e}")
            return ""
    
    @staticmethod
    def get_file_type(file_path: Path) -> str:
        """Determine file type"""
        suffix = file_path.suffix.lower()
        
        if suffix in FileUtils.SUPPORTED_TEXT_EXTENSIONS:
            return "text"
        elif suffix in FileUtils.SUPPORTED_IMAGE_EXTENSIONS:
            return "image"
        else:
            return "unknown"
    
    @staticmethod
    def is_supported_file(file_path: Path) -> bool:
        """Check if file is supported"""
        suffix = file_path.suffix.lower()
        return suffix in (FileUtils.SUPPORTED_TEXT_EXTENSIONS | 
                         FileUtils.SUPPORTED_IMAGE_EXTENSIONS)
    
    @staticmethod
    def get_mime_type(file_path: Path) -> str:
        """Get MIME type of file"""
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or "application/octet-stream"
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"