import base64
import hashlib
import os
import re
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


uuid_re = re.compile(r"<urn:uuid:([a-zA-Z0-9]{8}-?[a-zA-Z0-9]{4}-?[a-zA-Z0-9]{4}-?[a-zA-Z0-9]{4}-?[a-zA-Z0-9]{12})>")


def extract_uuid(uuid_urn: str) -> str:
    # Extract the UUID from the URN
    # "<urn:uuid:6a8657b3-84d0-45df-b4b2-5fb6eef55ee5>" -> "6a8657b384d045dfb4b25fb6eef55ee5"
    return uuid_re.sub("\\1", uuid_urn).replace("-", "")
