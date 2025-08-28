import shutil
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq
from datasets import Dataset, Features
from huggingface_hub import create_repo, list_repo_files

from c5.data_utils import upload_with_retry, yield_repo_parquet_files
from c5.script_utils import SCHEMA_NULLABLE


def is_fineweb(row: dict) -> bool:
    """
    Check if the document is in FineWeb(-2) based on the 'found_in_fw' field.
    """
    return row.get("found_in_fw", False) is True


def is_strict(row: dict) -> bool:
    """
    Check if the document is strict based on the 'found_in_fw' and 'license_abbr' fields.
    """
    return (
        (not row["license_disagreement"])
        and row["found_in_fw"]
        and "nc" not in row["license_abbr"]
        and row["license_abbr"] != "cc-unknown"
        and "wiki" not in row["url"]
    )


def main(
    version: Literal["fine", "strict"] = "fine",
    skip_dumps: list[str] = None,
    only_dumps: list[str] = None,
    num_proc: int | None = None,
    overwrite: bool = False,
):
    """
    Filter rows on whether or not they are in FineWeb(-2).

    Args:
        skip_dumps (list[str]): List of dumps to skip.
        only_dumps (list[str]): List of dumps to process.
        num_proc (int | None): Number of processes to use for multiprocessing.
    """
    if version not in ["fine", "strict"]:
        raise ValueError(f"Invalid version: {version}. Must be 'fine' or 'strict'.")

    tmp_dir = Path(__file__).parents[2] / "tmp" / "create_fine"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    skip_dumps = skip_dumps or []
    only_dumps = only_dumps or []

    orig_dataset_name = "BramVanroy/CommonCrawl-CreativeCommons"
    if version == "fine":
        filter_dataset_name = "BramVanroy/CommonCrawl-CreativeCommons-fine"
    else:
        filter_dataset_name = "BramVanroy/CommonCrawl-CreativeCommons-strict"

    create_repo(
        repo_id=filter_dataset_name,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    # [data/CC-MAIN-2025-05/spa/000_00001.parquet, ...]
    repo_fs = list_repo_files(orig_dataset_name, repo_type="dataset")
    if not repo_fs:
        raise ValueError(f"No files found in the original dataset {orig_dataset_name}.")

    crawls = {f.split("/")[1] for f in repo_fs if f.startswith("data/") and f.endswith(".parquet")}
    crawls = {c for c in crawls if c not in skip_dumps and (not only_dumps or c in only_dumps)}

    already_processed_remote_fs = list_repo_files(filter_dataset_name, repo_type="dataset")

    for crawl in crawls:
        crawl_tmp_dir = tmp_dir / crawl
        crawl_tmp_dir.mkdir(parents=True, exist_ok=True)
        print(f"Processing {crawl}...")
        for remote_parquet_uri, local_fname in yield_repo_parquet_files(
            orig_dataset_name,
            tmp_dir=str(crawl_tmp_dir),
            only_dumps=[crawl],
            skip_files_with_suffix=[] if overwrite else already_processed_remote_fs,
            skip_non_fineweb_dumps=True,
        ):
            features = Features.from_arrow_schema(SCHEMA_NULLABLE)
            ds = Dataset.from_parquet(local_fname, features=features)

            ds = ds.filter(
                is_fineweb if version == "fine" else is_strict,
                desc="Filtering",
                num_proc=num_proc,
            ).remove_columns("found_in_fw")

            if len(ds) > 0:
                ds.to_parquet(local_fname)
                # Test that the modified files can indeed be read correctly now with the right schema
                try:
                    _ = pq.read_table(local_fname, schema=SCHEMA_NULLABLE)
                except Exception as exc:
                    raise Exception(f"Could not read modified {local_fname}") from exc

                upload_with_retry(
                    repo_id=filter_dataset_name,
                    local_fname=local_fname,
                    remote_parquet_uri=remote_parquet_uri,
                )
            else:
                print(f"No items left after filtering in {local_fname}. Skipping upload...")

        shutil.rmtree(crawl_tmp_dir)
        print(f"Processed {crawl}.")
    print("Done")


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Filter rows on licenses, remove wiki, non-commercial, and unknown licenses",
    )
    cparser.add_argument(
        "--version",
        choices=["fine", "strict"],
        default="fine",
        help="Version of the dataset to create: 'fine' or 'strict'.",
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
    cparser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the remote dataset,"
        "i.e. reprocess the ones that are already in the HQ version",
    )

    cli_kwargs = vars(cparser.parse_args())
    main(**cli_kwargs)
