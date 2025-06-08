#!/usr/bin/env python3
import sys
import argparse
import concurrent.futures
import math
import re
from pathlib import Path
from typing import Optional, Tuple, List


PATTERN = re.compile(
    r"Total Runtime:\D*"
    r"(?:(\d+) days?)?\D*"
    r"(?:(\d+) hours?)?\D*"
    r"(?:(\d+) minutes?)?\D*"
    r"(?:(\d+) seconds?)?"
)


class RuntimeParseError(Exception):
    """Raised when a log file does not contain a valid Total Runtime line."""
    def __init__(self, file_path: Path):
        super().__init__(f"No valid runtime found in file: {file_path}")
        self.file_path = file_path


def parse_runtime_from_file(file_path: Path) -> Tuple[Path, int]:
    """Parse the runtime in seconds from a log file."""
    with file_path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = PATTERN.search(line)
            if match:
                days = int(match.group(1) or 0)
                hours = int(match.group(2) or 0)
                minutes = int(match.group(3) or 0)
                seconds = int(match.group(4) or 0)
                total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
                if total_seconds == 0:
                    raise ValueError(f"Got 0 seconds for {file_path} on this line: '{line}'. Extracted values: days: {days}, hours: {hours}, minutes: {minutes}, seconds: {seconds}.")
                return file_path, total_seconds
    raise RuntimeParseError(file_path)


def format_seconds(s: int) -> str:
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def print_progress(count: int, total: int, bar_len: int = 40) -> None:
    filled = int(bar_len * count / total)
    bar = '#' * filled + ' ' * (bar_len - filled)
    percent = int(100 * count / total)
    print(f"\r[{bar}] {percent:3d}% ({count}/{total})", end="", flush=True)


def get_log_files(log_dir: Path) -> List[Path]:
    return sorted(log_dir.rglob("*.log"))


def calculate_stats(runtimes: List[int]) -> dict:
    count = len(runtimes)
    runtimes_sorted = sorted(runtimes)
    total = sum(runtimes_sorted)
    mean = total / count
    min_ = runtimes_sorted[0]
    max_ = runtimes_sorted[-1]
    stdev = math.sqrt(sum((x - mean) ** 2 for x in runtimes_sorted) / count) if count > 1 else 0
    return {
        "count": count,
        "mean": mean,
        "min": min_,
        "max": max_,
        "stdev": stdev,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse and summarize runtimes from log files.")
    parser.add_argument("log_dir", type=Path, help="Directory containing *.log files")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: cpu count)")
    args = parser.parse_args()

    if not args.log_dir.is_dir():
        raise NotADirectoryError(f"'{args.log_dir}' is not a directory.")

    log_files = get_log_files(args.log_dir)
    total_files = len(log_files)
    if total_files == 0:
        raise FileNotFoundError(f"No log files found in {args.log_dir}")

    print(f"Found {total_files} log files.")
    print("Processing log files...")

    results: List[Optional[int]] = [None] * total_files
    errors: List[RuntimeParseError] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_idx = {
            executor.submit(parse_runtime_from_file, path): idx
            for idx, path in enumerate(log_files)
        }
        completed = 0
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                _, total_seconds = future.result()
                results[idx] = total_seconds
            except RuntimeParseError as e:
                errors.append(e)
            except Exception as e:
                # Propagate unknown errors immediately.
                raise
            completed += 1
            print_progress(completed, total_files)
    print()  # newline after progress bar

    if errors:
        print("\n[ERROR] The following files did not match the required pattern:\n", flush=True)
        for err in errors:
            print(f"  {err.file_path}")
        raise RuntimeParseError(errors[0].file_path)

    runtimes = [r for r in results if r is not None]
    if not runtimes:
        raise RuntimeError("No valid runtimes found in any log file.")

    stats = calculate_stats(runtimes)
    print(f"Files processed: {stats['count']}")
    print(f"Average: {format_seconds(round(stats['mean']))}")
    print(f"Min:     {format_seconds(stats['min'])}")
    print(f"Max:     {format_seconds(stats['max'])}")
    print(f"Stdev:   {format_seconds(round(stats['stdev']))}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFatal error: {e}", file=sys.stderr)
        raise

