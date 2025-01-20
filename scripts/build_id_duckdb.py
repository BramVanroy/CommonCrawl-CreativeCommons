import os
import re
from concurrent.futures import ProcessPoolExecutor

import duckdb
from datasets import Dataset, DatasetDict, IterableDatasetDict, concatenate_datasets, load_dataset


uuid_re = re.compile(r"<urn:uuid:([a-zA-Z0-9]{8}-?[a-zA-Z0-9]{4}-?[a-zA-Z0-9]{4}-?[a-zA-Z0-9]{4}-?[a-zA-Z0-9]{12})>")


def dataset_to_parquet(
    dataset_name: str,
    parquet_path: str,
    dataset_config: str | None = None,
    unique_column: str = "id",
    overwrite: bool = False,
    streaming: bool = False,
    num_loaders: int = 1,
    num_workers: int = 1,
) -> str:
    """
    Load a HuggingFace dataset and save one of its columns as a parquet file so that we can use it for
    querying other samples against it. The assumption is that the default `id` column is the WARC-Record-ID.

    Args:
        dataset_name: The name of the dataset
        parquet_path: The path to save the parquet file to
        dataset_config: The configuration of the dataset
        unique_column: The column to save. If this column is `id`, we remove the `urn:uuid:` prefix
        to make it compatible with DuckDB's UUID type
        overwrite: Whether to overwrite the parquet file if it already exists
        streaming: Whether to stream the dataset (likely slower but saves disk space)
        num_loaders: The number of parallel loaders to use. Only used when column is `id`.
        num_workers: The number of parallel workers to use.
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
        columns=[unique_column],
    )

    if isinstance(ds, DatasetDict) or isinstance(ds, IterableDatasetDict):
        ds = concatenate_datasets([ds[split] for split in list(ds.keys())])

    # "<urn:uuid:6a8657b3-84d0-45df-b4b2-5fb6eef55ee5>" -> "6a8657b3-84d0-45df-b4b2-5fb6eef55ee5"
    # = compatible with duckdb UUID type
    if unique_column == "id" and "fineweb" in dataset_name.lower():
        if not streaming:
            ds = ds.map(lambda idx: {"id": uuid_re.sub("\\1", idx)}, input_columns="id", num_proc=num_workers)
        else:
            ds = ds.map(lambda idx: {"id": uuid_re.sub("\\1", idx)}, input_columns="id")

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


if __name__ == "__main__":
    langs = ["nld_Latn", "fra_Latn", "deu_Latn", "spa_Latn", "ita_Latn", "fry_Latn", "afr_Latn"]

    def process_lang_fw2(dataset_name, lang):
        dataset_name = "HuggingFaceFW/fineweb-2"
        parquet_path = f"/home/ampere/vanroy/CommonCrawl-CreativeCommons/tmp/fw2-{lang}.parquet"
        duckdb_path = f"/home/ampere/vanroy/CommonCrawl-CreativeCommons/tmp/fw2-{lang}.duckdb"

        if not os.path.isfile(duckdb_path):
            parquet_path = dataset_to_parquet(
                dataset_name, parquet_path, lang, overwrite=False, streaming=True, num_loaders=8, num_workers=64
            )
            duckdb_path = parquet_to_duckdb(parquet_path, duckdb_path, overwrite=False)

    with ProcessPoolExecutor(max_workers=len(langs)) as p:
        p.map(process_lang_fw2, langs)
