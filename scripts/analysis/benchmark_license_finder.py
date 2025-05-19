import shutil
import time
import warnings
from os import PathLike
from pathlib import Path

from datatrove.pipeline.readers import WarcReader
from tqdm import tqdm


warnings.filterwarnings("ignore", category=DeprecationWarning)


from c5.components.annotators.license_annotator import (
    ParserException,
    _legacy_find_cc_licenses_in_html,
    find_cc_licenses_in_html,
)


FUNCS = {
    "improved": find_cc_licenses_in_html,
    "legacy": _legacy_find_cc_licenses_in_html,
}


def download_html(dump: str, tmp_dir: Path, limit: int) -> None:
    # Create a WarcReader instance to read WARC files from the specified S3 path
    warc_reader = WarcReader(
        f"s3://commoncrawl/crawl-data/{dump}/segments/",
        glob_pattern="*/warc/*",
        default_metadata={"dump": dump},
        limit=limit,
    )

    for doc_idx, doc in tqdm(enumerate(warc_reader.run()), desc="Downloading HTML files", total=limit):
        tmp_dir.joinpath(f"{doc_idx}.html").write_text(doc.text, encoding="utf-8")


def benchmark_license_finder(tmp_dir: Path, num_iters: int = 3, limit: int = 1000) -> None:
    """
    Benchmark `find_cc_licenses_in_html` vs `_legacy_find_cc_licenses_in_html` in performance speed and final number of licenses found.
    Use timeit to measure the time taken by each function for a given number of iterations.
    """
    html_files = list(tmp_dir.glob("*.html"))[:limit]
    html_strings = [file.read_text(encoding="utf-8") for file in html_files]
    print(f"Found {len(html_strings)} HTML files in {tmp_dir}")

    for func_name, func in FUNCS.items():
        total_time = 0
        times = []
        for iter_idx in range(num_iters):
            iter_total_time = 0
            iter_times = []
            total_licenses = 0
            for html_string in tqdm(
                html_strings, leave=False, desc=f"Processing {func_name} ({iter_idx + 1}/{num_iters})"
            ):
                start_time = time.perf_counter()

                try:
                    licenses = func(html_string)
                except ParserException:
                    pass

                total_licenses += len(licenses)
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                iter_times.append(elapsed_time)
                iter_total_time += elapsed_time

            avg_iter_time_per_doc = iter_total_time / len(html_strings) if html_strings else 0
            stdev_iter_time_per_doc = (
                (sum((x - avg_iter_time_per_doc) ** 2 for x in iter_times) / len(iter_times)) ** 0.5
                if iter_times
                else 0
            )
            print(
                f"{func_name} - Iter. {iter_idx + 1}/{num_iters}: Avg time/doc: {avg_iter_time_per_doc:.4f}s, stdev.: {stdev_iter_time_per_doc:.4f}, num. licenses found: {total_licenses}"
            )

            total_time += iter_total_time
            times.extend(iter_times)

        avg_time_per_doc = total_time / len(times) if times else 0
        stdev_per_doc = (sum((x - avg_time_per_doc) ** 2 for x in times) / len(times)) ** 0.5 if times else 0
        print(f"{func_name} - Avg. time/doc: {avg_time_per_doc:.4f} seconds, stdev.: {stdev_per_doc:.4f}")


def speed_benchmark(
    dump: str = "CC-MAIN-2019-30",
    tmp_dir: str | PathLike = Path(__file__).parents[2] / "tmp" / "benchmark",
    num_iters: int = 3,
    limit: int = 1000,
    overwrite_cache: bool = False,
) -> None:
    tmp_dir = Path(tmp_dir)

    if tmp_dir.exists() and tmp_dir.is_dir() and any(tmp_dir.iterdir()) and overwrite_cache:
        shutil.rmtree(tmp_dir)

    tmp_dir.mkdir(parents=True, exist_ok=True)

    num_files = len(list(Path(tmp_dir).glob("*.html")))
    extra_files_needed = limit - num_files
    if extra_files_needed > 0:
        print(f"Downloading HTML files from {dump}...")
        download_html(dump, tmp_dir, limit=limit)

    benchmark_license_finder(tmp_dir, num_iters=num_iters, limit=limit)


if __name__ == "__main__":
    # results_comparison(limit=10_000)
    # Results: yes, lxml is less robust against malformed HTML but that's okay - we assume that high-quality websites are preferred anyway
    # We lose 4 licenses out of 3514 (from 100_000 input docs) and all of them from malformed HTML, i.e. license AFTER closing body tag
    speed_benchmark(limit=100_000, num_iters=1)
    # improved - Iter. 1/1: Avg time/doc: 0.0015s, stdev.: 0.0155, num. licenses found: 3510
    # legacy - Iter. 1/1: Avg time/doc: 0.0291s, stdev.: 0.0492, num. licenses found: 3514
    # => improved is around 20x faster than legacy
