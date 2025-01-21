import gzip
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from datasets import Dataset, disable_caching, load_dataset
from huggingface_hub import create_repo
from tqdm import tqdm


disable_caching()


def get_data_robust(pfiles: list[Path]):
    """
    Given a set of .jsonl.gz files, this function reads them in a robust way, skipping incomplete lines,
    and yielding one sample at a time (parse-able JSON line).

    :param pfiles: A list of .jsonl.gz files
    :return: A generator yielding the contents of the files
    """
    with tqdm(total=len(pfiles), desc="Reading", unit="file") as pbar:
        for pfin in pfiles:
            if pfin.stat().st_size == 0:
                continue

            with gzip.open(pfin, "rt", encoding="utf-8") as f:
                while True:
                    try:
                        line = f.readline()
                        if not line:
                            break  # End of currently available content
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        # Handle partial or malformed JSON (incomplete writes)
                        continue
                    except EOFError:
                        # Handle unexpected EOF in gzip
                        break
            pbar.update(1)


def find_language_dirs(local_path: str) -> dict[str, list[Path]]:
    """
    Given a local path, this function finds all subdirectories that do not have subdirectories
    and groups them by their stem (language).

    :param local_path: The local path to the main output folder. Expected structure:
        local_path/{dump}/{language}/*.jsonl.gz
    :return: A dictionary of language: paths
    """
    plocal = Path(local_path).resolve()
    files = [pf.resolve() for pf in plocal.rglob("*.jsonl.gz") if pf.stat().st_size > 0]
    languages = {pf.parent.stem for pf in files}
    lang2files = {lang: [pf for pf in files if pf.parent.stem == lang] for lang in languages}

    return lang2files


def process_language_files(
    lang: str,
    files: list[Path],
    hf_repo: str,
    max_shard_size: str,
    public: bool,
    robust: bool,
    include_text: bool,
    num_cpus: int,
):
    """
    Given a language and a list of files, this function uploads the data to the HF hub.

    :param lang: The language
    :param files: The list of files
    :param hf_repo: The HF repo name
    :param max_shard_size: The maximum shard size
    :param public: Whether the repo should be public
    :param robust: Whether to read the JSONL files robustly, dropping incomplete lines
    :param include_text: Whether to include the 'text' column in the dataset
    :param num_cpus: Number of CPUs to use -- only used if robust is not set
    """
    if files:
        if robust:
            ds = Dataset.from_generator(get_data_robust, cache_dir=None, gen_kwargs={"pfiles": files})
        else:
            ds = load_dataset("json", data_files=[str(pf) for pf in files], split="train", num_proc=num_cpus)

        if not include_text:
            ds = ds.remove_columns("text")

        print(f"Uploading language {lang} to {hf_repo} in remote folder {lang} (config: {lang}; public: {public})")
        ds.push_to_hub(
            repo_id=hf_repo,
            config_name=lang,
            data_dir=lang,
            max_shard_size=max_shard_size,
            private=not public,
        )
        ds.cleanup_cache_files()
        del ds
        print(f"Finished uploading language {lang} to {hf_repo} in remote folder {lang} (config: {lang}; public: {public})")
    
    print(f"Nothing to do for language {lang}")

def main(
    local_path: str,
    hf_repo: str,
    max_shard_size: str = "500MB",
    public: bool = False,
    every: int | None = None,
    max_time: int | None = None,
    num_cpus: int | None = None,
    max_parallel_uploads: int = 8,
    robust: bool = False,
    include_text: bool = False,
):
    """
    Uploads a dataset of JSONL GZ files to the HF hub.

    :param local_path: The local path to the main output folder. Expected structure:
        local_path/{dump}/{language}/*.jsonl.gz
    :param hf_repo: The HF repo name
    :param max_shard_size: The maximum shard size
    :param public: Whether the repo should be public
    :param every: Upload every x minutes
    :param max_time: Maximum time to run in minutes. If 'every' is given and 'max_time'
    is not given, the script will run indefinitely
    :param num_cpus: Number of CPUs to use -- only used if robust is not set
    :param robust: Whether to read the JSONL files robustly, dropping incomplete lines
    """
    create_repo(repo_id=hf_repo, repo_type="dataset", private=not public, exist_ok=True)

    if max_time and not every:
        raise ValueError("If 'max_time' is set, 'every' must be set as well")

    if num_cpus is not None and num_cpus < 1:
        raise ValueError("num_cpus must be at least 1")

    num_cpus = num_cpus or max(os.cpu_count() - 1, 1)

    single_proc_fn = partial(
        process_language_files,
        hf_repo=hf_repo,
        max_shard_size=max_shard_size,
        public=public,
        robust=robust,
        include_text=include_text,
        num_cpus=num_cpus,
    )
    start_time = time.time()
    while True:
        lang2files = find_language_dirs(local_path)

        if not lang2files:
            print("No non-empty files found.")
        else:
            pool_size = min(max_parallel_uploads, len(lang2files))
            with ProcessPoolExecutor(max_workers=pool_size) as executor:
                executor.map(single_proc_fn, lang2files.keys(), lang2files.values())

        if every:
            time.sleep(every * 60)
            # If max_time is not given, we will run indefinitely
            if max_time and time.time() - start_time > max_time * 60:
                break
        else:
            break


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Upload outputs to the Hugging Face Hub in a specific repo. One config per language is created.",
    )
    cparser.add_argument("--local_path", type=str, required=True, help="Local path to upload")
    cparser.add_argument("--hf_repo", type=str, required=True, help="HF repo name")
    cparser.add_argument("--max_shard_size", type=str, help="Max shard size", default="500MB")
    cparser.add_argument("--public", action="store_true", help="Make the repo public", default=False)
    cparser.add_argument(
        "--every", type=int, help="Upload every x minutes. Requires 'max_time' to be set", default=None
    )
    cparser.add_argument("--max_time", type=int, help="Maximum time to run in minutes", default=None)
    cparser.add_argument("--num_cpus", type=int, help="Number of CPUs to use for reading a dataset", default=None)
    cparser.add_argument("--max_parallel_uploads", type=int, help="Maximum parallel uploads", default=8)
    cparser.add_argument(
        "--robust",
        action="store_true",
        help="Robustly read JSONL files, dropping incomplete lines (useful if process still running)",
        default=False,
    )
    cparser.add_argument(
        "--include_text",
        action="store_true",
        help="Include the 'text' column in the dataset",
        default=False,
    )

    cli_kwargs = vars(cparser.parse_args())
    main(**cli_kwargs)
