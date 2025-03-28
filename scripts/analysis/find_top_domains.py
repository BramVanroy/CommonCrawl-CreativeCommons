from pathlib import Path
from typing import Counter

import tldextract
from datasets import concatenate_datasets, load_dataset


no_cache_extract = tldextract.TLDExtract(cache_dir=None)

LANGUAGES = [
    "fry_Latn",
    "afr_Latn",
    "ita_Latn",
    "nld_Latn",
    "spa_Latn",
    "fra_Latn",
    "deu_Latn",
]

DUMPS = [
    "CC-MAIN-2019-30",
    "CC-MAIN-2020-05",
    "CC-MAIN-2023-06",
    "CC-MAIN-2024-51",
]


def find_crawl_non_cc(lang: str, num_proc: int | None = None, include_removed: bool = False, overwrite: bool = False):
    short_lang = lang.split("_")[0]
    pfout = Path(__file__).parent.parent / "tmp" / "not_found_in_creativecommons" / f"{short_lang}.txt"

    if pfout.exists():
        if overwrite:
            pfout.unlink()
        else:
            print(f"File {pfout} already exists. Skipping...")
            return

    datasets = []
    suffixes = ["", "_removed"] if include_removed else [""]
    for suffix in suffixes:
        cfg = f"{lang}{suffix}"
        try:
            ds = load_dataset("HuggingFaceFW/fineweb-2", name=cfg, columns=["id", "url", "dump", "file_path"])
        except ValueError:
            print(f"Dataset {cfg} not found")
            return
        else:
            # Concatenate splits
            ds = concatenate_datasets(list(ds.values()))
            datasets.append(ds)

    def get_domain(url):
        extracted = no_cache_extract(url)
        return {"domain": f"{extracted.domain}.{extracted.suffix}"}

    ds_fw2 = concatenate_datasets(datasets).map(get_domain, num_proc=num_proc, input_columns=["url"])
    all_domains = set(ds_fw2.unique("domain"))

    for dump in DUMPS:
        ds_cc = load_dataset(
            "BramVanroy/CommonCrawl-CreativeCommons", f"{dump}-{short_lang}", split="train", columns=["url"]
        )
        ds_cc = ds_cc.map(get_domain, num_proc=num_proc, input_columns=["url"])
        cc_domains = set(ds_cc.unique("domain"))
        # Remove the CC domains from the set if the are in it
        all_domains -= cc_domains
        print(f"CreativeCommons Dump {dump} - {short_lang}: {len(cc_domains):,} unique domains")
        print(f"Remaining non-CC unique domains: {len(all_domains):,}")

    ds_fw2_domain_not_in_cc = ds_fw2.filter(
        lambda domain: domain in all_domains, input_columns=["domain"], num_proc=num_proc
    )

    # Iterate over the dataset and count the domains rather than extracting the column to a Counter, to save memory
    domain_counts = Counter()
    for batch in ds_fw2_domain_not_in_cc.iter(batch_size=10_000):
        domains = batch["domain"]
        domain_counts.update(domains)

    # Sort by count
    sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)

    pfout.parent.mkdir(parents=True, exist_ok=True)
    pfout.write_text("\n".join([f"{domain}\t{count}" for domain, count in sorted_domains]) + "\n")

    # Top 2500
    pfout.with_name(f"{short_lang}-top2500.txt").write_text(
        "\n".join([f"{domain}\t{count}" for domain, count in sorted_domains[:2500]]) + "\n"
    )
    print(f"Finished {short_lang}!")


if __name__ == "__main__":
    for lang in LANGUAGES:
        find_crawl_non_cc(lang, num_proc=96)
