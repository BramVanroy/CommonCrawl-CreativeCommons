import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset, Features
from huggingface_hub import create_repo, list_repo_files, upload_file

from commoncrawl_cc_annotation.data_utils import yield_repo_parquet_files
from commoncrawl_cc_annotation.script_utils import SCHEMA_NULLABLE


def main(skip_dumps: list[str] = None, only_dumps: list[str] = None, num_proc: int | None = None, overwrite: bool = False):
    """
    Filter rows on licenses, remove wiki, non-commercial, and unknown licenses and
    on whether or not they are in FineWeb(-2). This is a post-processing step to
    create a recommended dataset for the CommonCrawl-CreativeCommons dataset.

    Args:
        skip_dumps (list[str]): List of dumps to skip.
        only_dumps (list[str]): List of dumps to process.
        num_proc (int | None): Number of processes to use for multiprocessing.
    """
    tmp_dir = Path(__file__).parents[2] / "tmp" / "create_recommended"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = str(tmp_dir)

    orig_dataset_name = "BramVanroy/CommonCrawl-CreativeCommons"
    filter_dataset_name = "BramVanroy/CommonCrawl-CreativeCommons-hq"

    create_repo(
        repo_id=filter_dataset_name,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )

    all_repo_files = list_repo_files(filter_dataset_name, repo_type="dataset")

    for remote_parquet_uri, local_fname in yield_repo_parquet_files(
        orig_dataset_name,
        tmp_dir=tmp_dir,
        only_dumps=only_dumps,
        skip_dumps=skip_dumps,
        skip_files_with_suffix=[] if overwrite else all_repo_files,
        skip_non_fineweb_dumps=True,
    ):
        features = Features.from_arrow_schema(SCHEMA_NULLABLE)
        ds = Dataset.from_parquet(local_fname, features=features)

        filter_kwargs = {
            "function": lambda x: (
                (not x["license_disagreement"])  # Only use pages with a consistent license
                and x["found_in_fw"]  # Only use pages that are in FineWeb(-2)
                and "nc" not in x["license_abbr"]  # Exclude non-commercial licenses
                and x["license_abbr"] != "cc-unknown"  # Exclude unknown licenses
                and "wiki" not in x["url"]  # Exclude Wiki-like pages (best to get those from a more reliable parser)
            ),
            "desc": "Filtering",
            "num_proc": num_proc,
        }

        ds = ds.filter(**filter_kwargs)

        if len(ds) > 0:
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
                        repo_id=filter_dataset_name,
                        commit_message="Filter on high-quality, high-certainty CC licenses",
                    )
                except Exception as exc:
                    num_retries -= 1
                    print(f"Error uploading {local_fname}: {exc}")
                    if not num_retries:
                        raise exc
                else:
                    break
        else:
            print(f"No items left after filtering in {local_fname}. Skipping upload...")

        os.unlink(local_fname)

    shutil.rmtree(tmp_dir)
    print("Done")


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Filter rows on licenses, remove wiki, non-commercial, and unknown licenses",
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
