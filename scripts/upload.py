import gzip
import json
import os
import time
from pathlib import Path

from datasets import Dataset, disable_caching, load_dataset
from huggingface_hub import create_repo
from tqdm import tqdm


disable_caching()


def get_data_robust(pfiles):
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


def main(
    local_path: str,
    hf_repo: str,
    config_name: str | None = None,
    max_shard_size: str = "500MB",
    public: bool = False,
    every: int | None = None,
    max_time: int | None = None,
    num_cpus: int | None = None,
    robust: bool = False,
    include_text: bool = False,
):
    """
    Uploads a dataset of JSONL GZ files to the HF hub.

    :param local_path: The local path to the folder to upload
    :param hf_repo: The HF repo name
    :param config_name: The HF repo config_name
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

    start_time = time.time()
    while True:
        files = list(Path(local_path).rglob("*.jsonl.gz"))
        # Filter empty files
        files = [f for f in files if f.stat().st_size > 0]
        if files:
            if robust:
                ds = Dataset.from_generator(get_data_robust, cache_dir=None, gen_kwargs={"pfiles": files})
            else:
                print(f"Loading dataset from {local_path} with {num_cpus} CPUs")
                ds = load_dataset("json", data_files=f"{local_path}/*.jsonl.gz", split="train", num_proc=num_cpus)

            if not include_text:
                ds = ds.remove_columns("text")

            print(
                f"Uploading folder {local_path} to {hf_repo}"
                f" in remote folder {config_name.replace('--', '/')}"
                f" (config: {config_name}; public: {public})"
            )
            ds.push_to_hub(
                repo_id=hf_repo,
                config_name=config_name if config_name else "default",
                data_dir=config_name.replace("--", "/") if config_name else None,
                max_shard_size=max_shard_size,
                private=not public,
                commit_message=f"Upload to {hf_repo} with config {config_name} and max_shard_size {max_shard_size}",
            )
            ds.cleanup_cache_files()

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
        description="Upload a folder to the HF hub. The uploaded folder will end up in the remote folder"
        " derived from the config_name by replacing '--' with '/', e.g. CC-MAIN-2019-30--base"
        " will be saved in CC-MAIN-2019-30/base.",
    )
    cparser.add_argument("--local_path", type=str, required=True, help="Local path to upload")
    cparser.add_argument("--hf_repo", type=str, required=True, help="HF repo name")
    cparser.add_argument("--config_name", type=str, help="HF repo config_name", default=None)
    cparser.add_argument("--max_shard_size", type=str, help="Max shard size", default="500MB")
    cparser.add_argument("--public", action="store_true", help="Make the repo public", default=False)
    cparser.add_argument(
        "--every", type=int, help="Upload every x minutes. Requires 'max_time' to be set", default=None
    )
    cparser.add_argument("--max_time", type=int, help="Maximum time to run in minutes", default=None)
    cparser.add_argument("--num_cpus", type=int, help="Number of CPUs to use", default=None)
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
