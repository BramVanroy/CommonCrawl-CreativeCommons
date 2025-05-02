import json
import shutil
import warnings
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files, upload_file
from huggingface_hub.errors import EntryNotFoundError
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import logging as transformers_logging


# Suppress the specific warning
warnings.filterwarnings("ignore", message=r".*Token indices sequence length is longer.*")
transformers_logging.set_verbosity_error()

CACHE_DIR = Path(__file__).parents[2] / "tmp" / "stats_download"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")


def count(texts: list[str]) -> dict[str, list[int]]:
    return {"num_tokens": [len(ids) for ids in tokenizer(texts)["input_ids"]]}


def get_stats(num_proc: int | None = None, force_overwrite: bool = False) -> None:
    dataset_name = "BramVanroy/CommonCrawl-CreativeCommons"
    files = list_repo_files(dataset_name, repo_type="dataset")

    configs = set()
    crawls = set()
    languages = set()
    for fname in files:
        if fname.startswith("data") and fname.endswith(".parquet"):
            fname = fname.replace("data/", "")
            # eg CC-MAIN-2025-05/spa
            cfg = fname.rsplit("/", 1)[0].replace("/", "-")
            crawl, lang = cfg.rsplit("-", 1)
            crawls.add(crawl)
            languages.add(lang)
            configs.add(cfg)

    configs = sorted(configs)
    crawls = sorted(crawls)
    languages = sorted(languages)
    print(f"Found {len(configs)} configs, {len(crawls)} crawls, and {len(languages)} languages.")

    results = {"crawls": {}, "languages": {}, "total": {"num_docs": 0, "num_tokens": 0}, "detailed": {}}
    if not force_overwrite:
        try:
            counts_fname = hf_hub_download(
                dataset_name,
                "counts.json",
                repo_type="dataset",
                force_download=True,
            )
        except EntryNotFoundError:
            pass
        else:
            # Makes sure that all keys are present in the dict
            results = {**results, **json.loads(Path(counts_fname).read_text(encoding="utf-8"))}

    for cfg in tqdm(configs):
        if cfg in results["detailed"]:
            print(f"Already counted {cfg}, skipping...")
            continue

        ds = load_dataset(
            "BramVanroy/CommonCrawl-CreativeCommons",
            cfg,
            split="train",
            cache_dir=CACHE_DIR / cfg,
        )
        ds = ds.select_columns(["text"])
        ds = ds.map(
            count, batched=True, num_proc=num_proc, input_columns=["text"], batch_size=15_000, desc=f"Counting {cfg}"
        )
        num_docs = len(ds)
        num_tokens = sum(ds["num_tokens"])
        results["detailed"][cfg] = {
            "num_docs": num_docs,
            "num_tokens": num_tokens,
        }

        # Specific aggregation
        crawl, lang = cfg.rsplit("-", 1)
        if crawl not in results["crawls"]:
            results["crawls"][crawl] = {"num_docs": 0, "num_tokens": 0}
        if lang not in results["languages"]:
            results["languages"][lang] = {"num_docs": 0, "num_tokens": 0}

        results["crawls"][crawl]["num_docs"] += num_docs
        results["crawls"][crawl]["num_tokens"] += num_tokens
        results["languages"][lang]["num_docs"] += num_docs
        results["languages"][lang]["num_tokens"] += num_tokens
        results["total"]["num_docs"] += num_docs
        results["total"]["num_tokens"] += num_tokens

        counts_fname = CACHE_DIR.joinpath("counts.json")
        counts_fname.write_text(
            json.dumps(results, indent=4),
            encoding="utf-8",
        )
        upload_file(
            repo_id=dataset_name,
            path_or_fileobj=counts_fname,
            path_in_repo="counts.json",
            repo_type="dataset",
        )
        shutil.rmtree(CACHE_DIR / cfg)
        print(f"Counted {cfg} ({results['detailed'][cfg]['num_docs']} docs, {results['detailed'][cfg]['num_tokens']} tokens)")


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(
        description="Count the number of docs and tokens in the Creative Commons dataset."
    )
    cparser.add_argument(
        "-j",
        "--num_proc",
        type=int,
        default=96,
        help="Number of processes to use for counting.",
    )
    cparser.add_argument(
        "-f",
        "--force_overwrite",
        action="store_true",
        help="By default already processed crawls will not be included. You can force an override here.",
    )
    cargs = cparser.parse_args()
    get_stats(num_proc=cargs.num_proc, force_overwrite=cargs.force_overwrite)
