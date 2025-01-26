import base64
import gzip
import hashlib
import json
import os
import re
from pathlib import Path

from tqdm import tqdm


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
    return uuid_re.sub("\\1", uuid_urn).replace("-", "")


def yield_jsonl_gz_data_robust(pfiles: list[Path], disable_tqdm: bool = False):
    """
    Given a set of .jsonl.gz files, this function reads them in a robust way, skipping incomplete lines,
    and yielding one sample at a time (parse-able JSON line).

    :param pfiles: A list of .jsonl.gz files
    :return: A generator yielding the contents of the files
    """
    with tqdm(total=len(pfiles), desc="Reading", unit="file", disable=disable_tqdm) as pbar:
        for pfin in pfiles:
            if pfin.stat().st_size == 0:
                continue

            with gzip.open(pfin, "rt", encoding="utf-8") as fhin:
                num_failures = 0
                while True:
                    try:
                        line = fhin.readline()
                        if not line:
                            break  # End of currently available content
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        # Handle partial or malformed JSON (incomplete writes)
                        num_failures += 1
                    except EOFError:
                        # Handle unexpected EOF in gzip
                        num_failures += 1
                        break
                if num_failures:
                    print(f"Skipped {num_failures:,} corrupt line(s) in {pfin}")
            pbar.update(1)
