import os

from datasets import load_dataset


def main(local_path: str, hf_repo: str, config_name: str | None = None, max_shard_size: str = "500MB", public: bool = False):
    """
    Uploads a given folder to the HF hub. The datatype is automatically inferred from the folder contents, e.g.
    .json or .jsonl.gz files.

    :param local_path: The local path to the folder to upload
    :param hf_repo: The HF repo name
    :param config_name: The HF repo config_name
    :param max_shard_size: The maximum shard size
    :param public: Whether the repo should be public
    """
    num_cpus = max(os.cpu_count() - 1, 1)

    print(f"Loading dataset from {local_path} with {num_cpus} CPUs")
    ds = load_dataset("json", data_files=f"{local_path}/*.jsonl.gz", split="train", num_proc=num_cpus)

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

    cli_kwargs = vars(cparser.parse_args())
    main(**cli_kwargs)
