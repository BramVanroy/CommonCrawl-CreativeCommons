import shutil
from pathlib import Path

import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files

from commoncrawl_cc_annotation.script_utils import SCHEMA_NULLABLE

"""DONE:
    "CC-MAIN-2019-30",
    "CC-MAIN-2020-05",
    "CC-MAIN-2023-06",
    "CC-MAIN-2024-51",

"""
CRAWLS = [
]

LANGUAGES = [
    "afr",
    "deu",
    "eng",
    "fra",
    "fry",
    "ita",
    "nld",
    "spa",
]

CACHE_DIR = Path(__file__).parents[1] / "tmp" / "hub_download_test"


def refresh_dir():
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def test_individual_parquet():
    all_repo_files = list_repo_files("BramVanroy/CommonCrawl-CreativeCommons", repo_type="dataset")
    parquet_files = [f for f in all_repo_files if f.endswith(".parquet")]

    langs_with_errors = {}
    crawls_with_errors = {}
    for remote_parquet_uri in parquet_files:
        pf = Path(remote_parquet_uri)
        lang = pf.parent.stem
        crawl = pf.parent.parent.stem
        num_retries = 3
        while num_retries:
            try:
                local_fname = hf_hub_download(
                    "BramVanroy/CommonCrawl-CreativeCommons",
                    filename=pf.name,
                    subfolder=pf.parent,
                    repo_type="dataset",
                    local_dir=CACHE_DIR / "parquet_tests",
                )
            except Exception as exc:
                num_retries -= 1
                print(f"Error downloading {pf}: {exc}", flush=True)
                if not num_retries:
                    raise Exception(f"Could not download {pf.name}") from exc
            else:
                break

        try:
            _ = pq.read_table(local_fname, schema=SCHEMA_NULLABLE)
        except Exception:
            # raise Exception(f"Could not read {local_fname}") from exc
            print(f"Could not read {local_fname}", flush=True)
            if lang not in langs_with_errors:
                langs_with_errors[lang] = 0
            langs_with_errors[lang] += 1
            if crawl not in crawls_with_errors:
                crawls_with_errors[crawl] = 0
            crawls_with_errors[crawl] += 1
        else:
            # print(f"Successfully read {local_fname}", flush=True)
            pass

    print("Parquet langs with errors:", langs_with_errors)
    print("Parquet crawls with errors:", crawls_with_errors)
    refresh_dir()


def test_hub_loading():
    for crawl in CRAWLS:
        try:
            ds = load_dataset(
                "BramVanroy/CommonCrawl-CreativeCommons", crawl, split="train", cache_dir=CACHE_DIR / crawl
            )
        except Exception as exc:
            raise Exception(f"Failed to load crawl {crawl}") from exc
        ds.cleanup_cache_files()
        refresh_dir()
    print("Successfully loaded all seperate crawls")

    # This might be getting too large, leading to memory-mapping errors -- so ignored
    # for lang in LANGUAGES:
    #     try:
    #         ds = load_dataset(
    #             "BramVanroy/CommonCrawl-CreativeCommons", lang, split="train", cache_dir=CACHE_DIR / lang
    #         )
    #     except Exception as exc:
    #         raise Exception(f"Failed to load language {lang}") from exc
    #     ds.cleanup_cache_files()
    #     refresh_dir()
    # print("Successfully loaded all seperate languages")

    for crawl in CRAWLS:
        for lang in LANGUAGES:
            config_name = f"{crawl}-{lang}"
            try:
                ds = load_dataset(
                    "BramVanroy/CommonCrawl-CreativeCommons",
                    config_name,
                    split="train",
                    cache_dir=CACHE_DIR / config_name,
                )
            except Exception as exc:
                raise Exception(f"Failed to load config {config_name}") from exc
            ds.cleanup_cache_files()
            refresh_dir()
    print("Successfully loaded all seperate crawl-lang configurations")


if __name__ == "__main__":
    test_individual_parquet()
    test_hub_loading()
