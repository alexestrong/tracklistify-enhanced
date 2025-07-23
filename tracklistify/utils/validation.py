"""
Input validation utilities for Tracklistify.
"""

# Standard library imports
from pathlib import Path
from typing import Optional

# Third-party imports
from yt_dlp import YoutubeDL, DownloadError

# Local/package imports
from .logger import get_logger

logger = get_logger(__name__)

# Supported audio file extensions
SUPPORTED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma'}


def validate_input(input_path: str) -> Optional[str]:
    """
    Validate input as either a local file path or URL.

    Args:
        input_path: Input path or URL to validate

    Returns:
        Validated path/URL if valid, None if invalid
    """
    # First, check if it's a local file path
    try:
        file_path = Path(input_path)
        
        # Check if it's an existing file
        if file_path.exists() and file_path.is_file():
            # Check if it has a supported audio extension
            if file_path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                logger.info(f"Validated local audio file: {input_path}")
                return str(file_path.resolve())  # Return absolute path
            else:
                logger.error(f"Unsupported audio format: {file_path.suffix}. Supported formats: {', '.join(SUPPORTED_AUDIO_EXTENSIONS)}")
                return None
        
        # If file doesn't exist but looks like a relative path with audio extension,
        # check if it exists relative to current working directory
        if not file_path.is_absolute() and file_path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
            # Try to resolve relative to current working directory
            abs_path = file_path.resolve()
            if abs_path.exists() and abs_path.is_file():
                logger.info(f"Validated local audio file: {abs_path}")
                return str(abs_path)
            else:
                logger.error(f"Audio file not found: {input_path}")
                return None
                
    except Exception as e:
        logger.debug(f"Path validation failed (will try as URL): {e}")
    
    # If it's not a local file, try to validate as URL
    logger.debug(f"Attempting to validate as URL: {input_path}")
    return validate_url(input_path)


def validate_url(url: str) -> Optional[str]:
    """
    Validate and clean a URL using yt-dlp.

    Args:
        url: Input URL to validate and clean

    Returns:
        Cleaned URL if valid, None if invalid
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "force_generic_extractor": True,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            if info_dict is not None:
                logger.info(f"Extracted info from URL: {info_dict.get('webpage_url')}")
                return info_dict.get("webpage_url", None)
            else:
                logger.error("Failed to extract info from URL")
                return None

    except DownloadError as e:
        logger.error(f"URL validation failed: {e}")
        return None


def is_youtube_url(url: str) -> bool:
    """
    Check if a URL is a valid YouTube URL.

    Args:
        url: URL to check

    Returns:
        bool: True if URL is a valid YouTube URL, False otherwise
    """
    if not url:
        return False

    cleaned_url = validate_input(url)
    if not cleaned_url:
        return False

    return "youtube.com/watch?v=" in cleaned_url


def is_mixcloud_url(url: str) -> bool:
    """
    Check if a URL is a valid Mixcloud URL.

    Args:
        url: URL to check

    Returns:
        bool: True if URL is a valid Mixcloud URL, False otherwise
    """
    if not url:
        return False

    cleaned_url = validate_input(url)
    if not cleaned_url:
        return False

    return "mixcloud.com/" in cleaned_url
