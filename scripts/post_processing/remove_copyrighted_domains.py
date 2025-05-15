import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import tldextract
from datasets import Features, load_dataset
from datasets.arrow_dataset import Dataset
from huggingface_hub.hf_api import upload_file

from c5.data_utils import yield_repo_parquet_files
from c5.script_utils import SCHEMA_NULLABLE


no_cache_extract = tldextract.TLDExtract(cache_dir=None)


def main(
    dataset_name: str = "BramVanroy/CommonCrawl-CreativeCommons",
    skip_dumps: list[str] = None,
    only_dumps: list[str] = None,
    num_proc: int | None = None,
):
    """
    Remove rows with domains that are known to be C&D'd from the CommonCrawl-CreativeCommons dataset.
    These domains are saved in BramVanroy/finewebs-copyright-domains.

    Args:
        dataset_name (str): The name of the dataset to process.
        skip_dumps (list[str]): List of dumps to skip.
        only_dumps (list[str]): List of dumps to process.
        num_proc (int | None): Number of processes to use for multiprocessing.
    """
    ds_domains = load_dataset("BramVanroy/finewebs-copyright-domains", split="train")
    remove_domains = set(ds_domains.unique("domain"))
    del ds_domains
    print("Domains to remove:")
    print(remove_domains)

    tmp_dir = Path(__file__).parents[2] / "tmp" / "remove_domains"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = str(tmp_dir)

    for remote_parquet_uri, local_fname in yield_repo_parquet_files(
        dataset_name,
        tmp_dir=tmp_dir,
        only_dumps=only_dumps,
        skip_dumps=skip_dumps,
    ):
        features = Features.from_arrow_schema(SCHEMA_NULLABLE)
        ds = Dataset.from_parquet(local_fname, features=features)
        num_items = len(ds)

        def get_domain(url):
            extracted = no_cache_extract(url)
            return {"domain": f"{extracted.domain}.{extracted.suffix}"}

        ds = ds.map(get_domain, num_proc=num_proc, input_columns=["url"], desc="Extracting domains")
        ds = (
            ds.filter(
                lambda domain: domain not in remove_domains,
                input_columns=["domain"],
                desc="Filtering out removed domains",
                num_proc=num_proc,
            )
            .remove_columns("domain")
            .cast(features=features)
        )
        new_num_items = len(ds)

        if new_num_items < num_items:
            print(f"Removed {num_items - new_num_items} rows with removed domains in {local_fname}")
            ds.to_parquet(local_fname)
            # Test that the modified files can indeed be read correctly now with the right schema
            try:
                _ = pq.read_table(local_fname, schema=SCHEMA_NULLABLE)
            except Exception as exc:
                raise Exception(f"Could not read modified {local_fname}") from exc

            num_retries = 3
            while num_retries:
                try:
                    upload_file(
                        path_or_fileobj=local_fname,
                        path_in_repo=remote_parquet_uri,
                        repo_type="dataset",
                        repo_id=dataset_name,
                        commit_message="Remove rows with C&D domains",
                    )
                except Exception as exc:
                    num_retries -= 1
                    print(f"Error uploading {local_fname}: {exc}")
                    if not num_retries:
                        raise exc
                else:
                    break
        else:
            print(f"No rows with removed domains in {local_fname}")

        os.unlink(local_fname)

    shutil.rmtree(tmp_dir)
    print("Done")


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Remove rows with domains that are known to be C&D'd from the CommonCrawl-CreativeCommons dataset."
        " This is part of the local and slurm pipelines but you can use this script if you want to run it manually.",
    )
    cparser.add_argument(
        "--skip-dumps",
        nargs="+",
        default=[],
        help="Dumps to skip (e.g. CC-MAIN-2021-04)",
    )
    cparser.add_argument(
        "--only-dumps",
        nargs="+",
        default=[],
        help="Only process these dumps (e.g. CC-MAIN-2021-04)",
    )
    cparser.add_argument(
        "-j",
        "--num-proc",
        type=int,
        default=None,
        help="Number of processes to use for multiprocessing (default: None)",
    )

    cli_kwargs = vars(cparser.parse_args())
    main(**cli_kwargs)
