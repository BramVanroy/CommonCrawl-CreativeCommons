import os
from pathlib import Path
from shutil import rmtree
from time import sleep
from typing import Callable

import duckdb
from datasets import (
    DatasetDict,
    IterableDatasetDict,
    concatenate_datasets,
    disable_caching,
    get_dataset_config_names,
    load_dataset,
)
from huggingface_hub import list_repo_files, upload_file
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.hf_api import create_repo, repo_exists

from commoncrawl_cc_annotation.utils import extract_uuid


# Disable chacing to save space
disable_caching()


def dataset_to_duckdb(
    dataset_name: str,
    duckdb_path: str,
    id_prep_func: Callable = None,
    dataset_config: str | None = None,
    cache_dir: str | None = None,
    clear_cache_dir: bool = True,
    overwrite: bool = False,
    streaming: bool = False,
    num_loaders: int = 1,
    num_workers: int = 1,
) -> bool:
    """
    Load a HuggingFace dataset and save one of its columns as a parquet file so that we can use it to
    build a DuckDB database from.

    Args:
        dataset_name: The name of the dataset
        duckdb_path: The path to save the duckdb file to
        unique_column_func: A function to apply to each sample to add the unique column, or the name of the unique column
        dataset_config: The configuration of the dataset
        overwrite: Whether to overwrite the parquet file if it already exists
        streaming: Whether to stream the dataset (likely slower but saves disk space)
        num_loaders: The number of parallel loaders to use. 
        num_workers: The number of parallel workers to use when applying the unique_column_func if it is a Callable
        only_load_columns: The columns to load from the dataset. Can improve the download time by only downloading these columns

    Returns:
        Whether the DuckDB file was successfully built
    """
    if streaming:
        raise NotImplementedError("Streaming is not yet supported")

    if os.path.isfile(duckdb_path) and os.path.getsize(duckdb_path) > 0:
        if overwrite:
            os.remove(duckdb_path)
        else:
            print(f"DuckDB file already exists at {duckdb_path}. Skipping...")
            return duckdb_path
    else:
        os.makedirs(os.path.dirname(duckdb_path), exist_ok=True)

    print(f"Starting to build DuckDB file at {duckdb_path}")
    try:
        ds = load_dataset(
            dataset_name,
            dataset_config,
            streaming=streaming,
            num_proc=num_loaders if not streaming else None,
            # Only works when the origin files are parquet.
            columns=["dump", "id"],
            cache_dir=cache_dir,
        )
    except (TypeError, ValueError):
        try:
            ds = load_dataset(
                dataset_name,
                dataset_config,
                streaming=streaming,
                num_proc=num_loaders if not streaming else None,
            )
        except Exception:
            print(f"Failed to load dataset {dataset_name} with config {dataset_config}")
            return False

    if isinstance(ds, DatasetDict) or isinstance(ds, IterableDatasetDict):
        ds = concatenate_datasets([ds[split] for split in list(ds.keys())])

    ds = ds.remove_columns([c for c in ds.column_names if c not in ("dump", "id")])

    if id_prep_func:
        if streaming:
            ds = ds.map(lambda sample: {"id": id_prep_func(sample)})
        else:
            ds = ds.map(lambda sample: {"id": id_prep_func(sample)}, num_proc=num_workers)

    farrow = str(Path(duckdb_path).with_suffix(".parquet").resolve())
    ds.to_parquet(farrow)
    del ds
       
    con = duckdb.connect(duckdb_path)
    if dataset_name == "HuggingFaceFW/fineweb-2":         
        # Create table with two columns: dump and id, and add a primary key on the composite
        con.execute(f"""
            CREATE OR REPLACE TABLE data (
                dump STRING,
                id UUID,
                PRIMARY KEY (dump, id)
            );
            INSERT INTO data
                SELECT DISTINCT dump, id FROM '{farrow}';
            """)
    else:
        # Because we create one duckdb for each dump, we don't need the dump column in the table
        # Create table with one column: id, and add a primary key on it
        con.execute(f"""
            CREATE OR REPLACE TABLE data (
                id UUID PRIMARY KEY
            );
            INSERT INTO data
                SELECT DISTINCT id FROM '{farrow}';
            """)

    row_count = con.execute("SELECT COUNT(*) FROM data").fetchone()[0]
    if row_count == 0:
        raise ValueError(f"No rows were inserted into the DuckDB database at {duckdb_path}")

    print(f"The table in {farrow} has {row_count:,} rows.")

    os.remove(farrow)

    if cache_dir and clear_cache_dir:
        rmtree(cache_dir)

    return True


def fw_prep_func(sample: dict):
    # "<urn:uuid:6a8657b3-84d0-45df-b4b2-5fb6eef55ee5>" -> "6a8657b384d045dfb4b25fb6eef55ee5"
    # = compatible with duckdb UUID type
    return extract_uuid(sample["id"])


KEEP_LOCAL = [
    "fry_Latn",
    "afr_Latn",
    "ita_Latn",
    "nld_Latn",
    "spa_Latn",
    "fra_Latn",
    "deu_Latn",
]


def build_all_fw2_dbs(overwrite: bool = False):
    dataset_name = "HuggingFaceFW/fineweb-2"
    config_names = [cfg for cfg in get_dataset_config_names(dataset_name) if "removed" not in cfg]
    existing_files_in_repo = list_repo_files(repo_id="BramVanroy/fineweb-2-duckdbs", repo_type="dataset")

    language_success = {}
    for lang in config_names:
        local_duckdb_path = f"/home/ampere/vanroy/CommonCrawl-CreativeCommons/duckdbs/fineweb-2/fw2-{lang}.duckdb"
        path_in_repo = Path(local_duckdb_path).name
        exists_in_repo = path_in_repo in existing_files_in_repo

        if overwrite or (not os.path.isfile(local_duckdb_path) and not exists_in_repo):
            print(f"Buidling DuckDB for {lang}")
            lang_success = dataset_to_duckdb(
                "HuggingFaceFW/fineweb-2",
                local_duckdb_path,
                # For fineweb we still have to extract the UUID from the `id` column
                id_prep_func=fw_prep_func,
                dataset_config=lang,
                overwrite=False,
                streaming=False,
                num_loaders=None,
                num_workers=64,
            )
            language_success[lang] = lang_success
        else:
            language_success[lang] = True

        if os.path.isfile(local_duckdb_path) and (overwrite or not exists_in_repo):
            print(f"Uploading {local_duckdb_path}")
            num_retries = 3
            while num_retries:
                try:
                    upload_file(
                        path_or_fileobj=local_duckdb_path,
                        path_in_repo=path_in_repo,
                        repo_id="BramVanroy/fineweb-2-duckdbs",
                        repo_type="dataset",
                    )
                except HfHubHTTPError as exc:
                    num_retries -= 1
                    if num_retries == 0:
                        raise exc
                    else:
                        sleep(60 / num_retries)
                else:
                    break

            if lang not in KEEP_LOCAL:
                os.remove(local_duckdb_path)

            sleep(30)

    print("Failed languages:")
    for lang, success in language_success.items():
        if not success:
            print(f"- {lang}")


def build_fw_dbs(overwrite: bool = False):
    dataset_name = "HuggingFaceFW/fineweb"
    dump_names = [cfg for cfg in get_dataset_config_names(dataset_name) if "sample" not in cfg and cfg != "default"]
    priority_dumps = ["CC-MAIN-2019-30", "CC-MAIN-2020-05", "CC-MAIN-2021-04", "CC-MAIN-2022-05", "CC-MAIN-2023-06", "CC-MAIN-2024-51", "CC-MAIN-2025-05"]
    priority_dumps = [d for d in priority_dumps if d in dump_names]
    dump_names = priority_dumps + [d for d in dump_names if d not in priority_dumps]

    print("Dump names")
    print(dump_names)

    if not repo_exists("BramVanroy/fineweb-duckdbs", repo_type="dataset"):
        create_repo("BramVanroy/fineweb-duckdbs", repo_type="dataset")

    existing_files_in_repo = list_repo_files(repo_id="BramVanroy/fineweb-duckdbs", repo_type="dataset")

    dump_success_result = {}
    for dump in dump_names:
        local_duckdb_path = f"/home/ampere/vanroy/CommonCrawl-CreativeCommons/duckdbs/fineweb/fw-{dump}.duckdb"
        path_in_repo = Path(local_duckdb_path).name
        exists_in_repo = path_in_repo in existing_files_in_repo

        if overwrite or (not os.path.isfile(local_duckdb_path) and not exists_in_repo):
            print(f"Buidling DuckDB for {dump}")
            dump_success = dataset_to_duckdb(
                "HuggingFaceFW/fineweb",
                local_duckdb_path,
                # For fineweb we still have to extract the UUID from the `id` column
                id_prep_func=fw_prep_func,
                dataset_config=dump,
                overwrite=False,
                streaming=False,
                num_loaders=4,
                num_workers=64,
                cache_dir=f"/home/ampere/vanroy/CommonCrawl-CreativeCommons/tmp/fw/{dump}",
                clear_cache_dir=True,
            )
            dump_success_result[dump] = dump_success
        else:
            dump_success_result[dump] = True

        if os.path.isfile(local_duckdb_path) and (overwrite or not exists_in_repo):
            print(f"Uploading {local_duckdb_path}")
            num_retries = 3
            while num_retries:
                try:
                    upload_file(
                        path_or_fileobj=local_duckdb_path,
                        path_in_repo=path_in_repo,
                        repo_id="BramVanroy/fineweb-duckdbs",
                        repo_type="dataset",
                    )
                except HfHubHTTPError as exc:
                    num_retries -= 1
                    if num_retries == 0:
                        raise exc
                    else:
                        sleep(60 / num_retries)
                else:
                    break

            if dump not in KEEP_LOCAL:
                os.remove(local_duckdb_path)

            sleep(30)

    print("Failed languages:")
    for dump, success in dump_success_result.items():
        if not success:
            print(f"- {dump}")

if __name__ == "__main__":
    # build_all_fw2_dbs()
    build_fw_dbs()

