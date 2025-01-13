import os
from pathlib import Path


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
