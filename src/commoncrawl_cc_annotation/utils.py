import base64
import hashlib
import os
from pathlib import Path


# Root directory of this library
PROJECT_ROOT = Path(__file__).parents[2]


def print_system_stats():
    """
    Print out the number of CPU cores on the system as well as the available memory.
    """
    print(f"Number of CPU cores: {os.cpu_count()}")

    try:
        print(f"Available memory: {os.popen('free -h').read()}")
    except Exception:
        pass


def generate_base64_hash(input_string: str) -> str:
    sha256_hash = hashlib.sha256(input_string.encode()).digest()
    return base64.urlsafe_b64encode(sha256_hash).decode("utf-8")
