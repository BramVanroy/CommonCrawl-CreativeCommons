import shutil
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


CRAWLS = [
    "CC-MAIN-2019-30",
    "CC-MAIN-2020-05",
    # "CC-MAIN-2023-06",
    # "CC-MAIN-2024-51",
]

IGNORE_CRAWLS = []

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

CACHE_DIR = Path(__file__).parents[1] / "tmp" / "stats_download"


def refresh_dir():
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_stats(only_found_in_fw: bool = False):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")

    def count_num_tokens(texts: list[str]) -> dict[str, list[int]]:
        return {"num_tokens": [len(ids) for ids in tokenizer(texts)["input_ids"]]}

    num_docs = {crawl: {lang: 0 for lang in LANGUAGES} for crawl in CRAWLS}
    num_tokens = {crawl: {lang: 0 for lang in LANGUAGES} for crawl in CRAWLS}
    for crawl in CRAWLS:
        if crawl in IGNORE_CRAWLS:
            continue
        for lang in LANGUAGES:
            try:
                split_name = f"{crawl}-{lang}"
                ds = load_dataset(
                    "BramVanroy/CommonCrawl-CreativeCommons", split_name, split="train", cache_dir=CACHE_DIR
                )
            except Exception as exc:
                raise Exception(f"Failed to load crawl {crawl}") from exc

            if only_found_in_fw:
                ds = ds.filter(lambda x: x["found_in_fw"], num_proc=64)

            if len(ds) > 0:
                ds = ds.map(count_num_tokens, batched=True, num_proc=32, input_columns="text")
                num_docs[crawl][lang] = len(ds)
                num_tokens[crawl][lang] = sum(ds["num_tokens"])
            else:
                num_docs[crawl][lang] = 0
                num_tokens[crawl][lang] = 0

            ds.cleanup_cache_files()

    # Adding `[tab]` to make it easier to copy-paste and replace, rather than the terminal messing up the formatting`
    print_table = "crawl[tab]num_docs_afr[tab]num_docs_deu[tab]num_docs_eng[tab]num_docs_fra[tab]num_docs_fry[tab]num_docs_ita[tab]num_docs_nld[tab]num_docs_spa[tab]total_num_docs[tab]num_toks_afr[tab]num_toks_deu[tab]num_toks_eng[tab]num_toks_fra[tab]num_toks_fry[tab]num_toks_ita[tab]num_toks_nld[tab]num_toks_spa[tab]total_num_toks\n"
    for crawl in CRAWLS:
        print_table += f"{crawl}[tab]"
        for dtype in ("docs", "tokens"):
            total = 0
            for lang in LANGUAGES:
                print_table += (
                    f"{num_docs[crawl][lang]}[tab]" if dtype == "docs" else f"{num_tokens[crawl][lang]}[tab]"
                )
                total += num_docs[crawl][lang] if dtype == "docs" else num_tokens[crawl][lang]
            
            print_table += f"{total}[tab]"
        print_table += "\n"
    print(print_table)


if __name__ == "__main__":
    get_stats()
    # get_stats(found_in_fw=True)
