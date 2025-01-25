import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable

import duckdb
from datasets import Dataset, DatasetDict, IterableDatasetDict, concatenate_datasets, load_dataset

from commoncrawl_cc_annotation.utils import extract_uuid


def dataset_to_parquet(
    dataset_name: str,
    parquet_path: str,
    unique_column_func_or_name: Callable | str,
    dataset_config: str | None = None,
    overwrite: bool = False,
    streaming: bool = False,
    num_loaders: int = 1,
    num_workers: int = 1,
    only_load_columns: list[str] | None = None,
) -> str:
    """
    Load a HuggingFace dataset and save one of its columns as a parquet file so that we can use it to
    build a DuckDB database from.

    Args:
        dataset_name: The name of the dataset
        parquet_path: The path to save the parquet file to
        unique_column_func: A function to apply to each sample to add the unique column, or the name of the unique column
        dataset_config: The configuration of the dataset
        overwrite: Whether to overwrite the parquet file if it already exists
        streaming: Whether to stream the dataset (likely slower but saves disk space)
        num_loaders: The number of parallel loaders to use. Only used when column is `id`.
        num_workers: The number of parallel workers to use when applying the unique_column_func if it is a Callable
        only_load_columns: The columns to load from the dataset. Can improve the download time by only downloading these columns
    """

    if os.path.isfile(parquet_path):
        if overwrite:
            os.remove(parquet_path)
        else:
            print(f"Parquet file already exists at {parquet_path}. Skipping...")
            return parquet_path

    ds = load_dataset(
        dataset_name,
        dataset_config,
        streaming=streaming,
        num_proc=num_loaders if not streaming else None,
        # Only works when the origin files are parquet. So still have to do `remove_columns` later on
        columns=only_load_columns,
    )

    if isinstance(ds, DatasetDict) or isinstance(ds, IterableDatasetDict):
        ds = concatenate_datasets([ds[split] for split in list(ds.keys())])

    if isinstance(unique_column_func_or_name, str):
        ds = ds.remove_columns([c for c in ds.column_names if c != unique_column_func_or_name])
    else:
        if streaming:
            ds = ds.map(unique_column_func_or_name, remove_columns=ds.column_names)
        else:
            ds = ds.map(unique_column_func_or_name, num_proc=num_workers, remove_columns=ds.column_names)

    if streaming:

        def generate_from_iterable_ds(_ds):
            yield from _ds

        ds = Dataset.from_generator(
            generate_from_iterable_ds, gen_kwargs={"_ds": ds}, features=ds.features, num_proc=num_loaders
        )

    ds.to_parquet(parquet_path)

    ds.cleanup_cache_files()
    del ds

    return parquet_path


def parquet_to_duckdb(
    parquet_path: str, duckdb_path: str, unique_column: str = "id", overwrite: bool = False, column_type: str = "UUID"
) -> str:
    """
    For many repeated queries, using duckdb over parquet may be worthwhile: https://duckdb.org/docs/guides/performance/file_formats.html
    so we convert the parquet file to a duckdb file, even though one could also directly query the parquet files. DuckDB is faster.

    Args:
        parquet_path: Path to the parquet input file
        duckdb_path: Path to the duckdb output file
        unique_column: The column to use as the primary key
        overwrite: Whether to overwrite the duckdb file if it already exists
        column_type: The type of the unique column, must be a valid DuckDB data type

    Returns:
        Path to the duckdb file
    """
    if os.path.isfile(duckdb_path):
        if overwrite:
            os.remove(duckdb_path)
        else:
            print(f"DuckDB file already exists at {duckdb_path}. Skipping...")
            return duckdb_path

    con = duckdb.connect(duckdb_path)
    # Create table with a single unique column and add a primary key constraint for speed
    query = f"""
        CREATE OR REPLACE TABLE data (
            {unique_column} {column_type} PRIMARY KEY
        );
        INSERT INTO data
        SELECT DISTINCT {unique_column} FROM '{parquet_path}'
        """

    con.execute(query)

    con.close()
    return duckdb_path


def fw2_prep_func(sample: dict):
    # "<urn:uuid:6a8657b3-84d0-45df-b4b2-5fb6eef55ee5>" -> "6a8657b3-84d0-45df-b4b2-5fb6eef55ee5"
    # = compatible with duckdb UUID type
    return {"id": extract_uuid(sample["id"])}


if __name__ == "__main__":
    langs = ["nld_Latn", "fra_Latn", "deu_Latn", "spa_Latn", "ita_Latn", "fry_Latn", "afr_Latn"]

    for dataset_name, subdir, prefix in [
        ("HuggingFaceFW/fineweb-2", "fineweb-2", "fw2"),
        ("HPLT/HPLT2.0_cleaned", "HPLT2.0_cleaned", "hplt2"),
    ]:

        def process_lang(lang):
            parquet_path = f"/home/ampere/vanroy/CommonCrawl-CreativeCommons/duckdbs/{subdir}/{prefix}-{lang}.parquet"
            duckdb_path = f"/home/ampere/vanroy/CommonCrawl-CreativeCommons/duckdbs/{subdir}/{prefix}-{lang}.duckdb"

            if not os.path.isfile(duckdb_path):
                parquet_path = dataset_to_parquet(
                    dataset_name,
                    parquet_path,
                    # For fineweb we still have to extract the UUID from the `id` column, for hplt we can use it as-is
                    unique_column_func_or_name=fw2_prep_func if dataset_name == "HuggingFaceFW/fineweb-2" else "id",
                    dataset_config=lang,
                    overwrite=True,
                    streaming=True,
                    num_loaders=8,
                    num_workers=64,
                    only_load_columns=["id"],
                )
                duckdb_path = parquet_to_duckdb(parquet_path, duckdb_path, overwrite=False)

        with ProcessPoolExecutor(max_workers=len(langs)) as p:
            p.map(process_lang, langs)
