from datasets import Dataset, load_dataset


def main(
    languages: list[str],
    max_samples_per_lang: int = 5000,
):
    ds = load_dataset("BramVanroy/CommonCrawl-CreativeCommons-fine", split="train").shuffle(seed=42)
    ds = ds.filter(lambda text: text and len(text.split()) >= 40, num_proc=96, input_columns=["text"])
    ds = ds.add_column("index", list(range(len(ds))))
    uniq_langs = ds.unique("language")

    sample_idxs = []
    counts = {lang: 0 for lang in uniq_langs if lang in languages}
    for lang in uniq_langs:
        lang_ds = ds.filter(lambda col_lang: col_lang == lang, input_columns="language", num_proc=96)
        target_n = min(len(lang_ds), max_samples_per_lang)
        sample_idxs.extend(lang_ds["index"][:target_n])
        counts[lang] = target_n

    print(counts)

    # Convert number to k, M
    if max_samples_per_lang >= 1_000_000:
        num_str = f"{max_samples_per_lang // 1_000_000}M"
    elif max_samples_per_lang >= 1_000:
        num_str = f"{max_samples_per_lang // 1_000}k"
    else:
        num_str = str(max_samples_per_lang)

    sampled_ds = ds.select(sample_idxs)
    print(sampled_ds)
    sampled_ds.push_to_hub(f"BramVanroy/CommonCrawl-CreativeCommons-fine-language-{num_str}", private=True)


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(
        description="Create a language balanced subset of the CommonCrawl-CreativeCommons-fine dataset by sampling a maximum number of random samples per language.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    cparser.add_argument(
        "-l",
        "--languages",
        type=str,
        nargs="+",
        default=[
            "eng",
            "deu",
            "fra",
            "spa",
            "ita",
            "nld",
            "afr",
            "fry",
        ],
        help="List of languages to include in the subset (ISO 639-1 codes)",
    )
    cparser.add_argument(
        "-n", "--max-samples-per-lang", type=int, default=5000, help="Maximum number of samples per language"
    )
    args = cparser.parse_args()
    main(**vars(args))
