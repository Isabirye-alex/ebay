# utils/helpers.py

import os


def validate_file_path(file_path: str) -> str:
    """
    Validate that a file path exists.

    Args:
        file_path (str): Path to file

    Returns:
        str: Validated file path

    Raises:
        FileNotFoundError: If file does not exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return file_path
