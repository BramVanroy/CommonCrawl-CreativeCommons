import os
from pathlib import Path
from typing import Callable

import duckdb
from datasets import DatasetDict, IterableDatasetDict, concatenate_datasets, load_dataset
from tqdm import tqdm

from commoncrawl_cc_annotation.utils import extract_uuid


def dataset_to_duckdb(
    dataset_name: str,
    duckdb_path: str,
    id_prep_func: Callable = None,
    dataset_config: str | None = None,
    overwrite: bool = False,
    streaming: bool = False,
    num_loaders: int = 1,
    num_workers: int = 1,
) -> str:
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
        num_loaders: The number of parallel loaders to use. Only used when column is `id`.
        num_workers: The number of parallel workers to use when applying the unique_column_func if it is a Callable
        only_load_columns: The columns to load from the dataset. Can improve the download time by only downloading these columns
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
    ds = load_dataset(
        dataset_name,
        dataset_config,
        streaming=streaming,
        num_proc=num_loaders if not streaming else None,
        # Only works when the origin files are parquet.
        columns=["dump", "id"],
    )

    if isinstance(ds, DatasetDict) or isinstance(ds, IterableDatasetDict):
        ds = concatenate_datasets([ds[split] for split in list(ds.keys())])

    ds = ds.remove_columns([c for c in ds.column_names if c not in ("dump", "id")])

    if id_prep_func:
        if streaming:
            ds = ds.map(lambda sample: {"id": id_prep_func(sample)})
        else:
            ds = ds.map(lambda sample: {"id": id_prep_func(sample)}, num_proc=num_workers)

    pfarrow = Path(duckdb_path).with_suffix(".parquet").resolve()
    ds = ds.to_parquet(pfarrow)
    print(f"Finished writing parquet file to {pfarrow}")

    if streaming:
        pass
    else:
        con = duckdb.connect(duckdb_path)
        # Create table with two columns: dump and id, and add a primary key on the composite
        con.execute(f"""
            CREATE OR REPLACE TABLE data (
                dump STRING,
                id UUID,
                PRIMARY KEY (dump, id)
            );
            INSERT INTO data
                SELECT DISTINCT dump, id FROM '{str(pfarrow)}';
            """)

        con.commit()
        con.close()

    pfarrow.unlink()

    return duckdb_path


def fw2_prep_func(sample: dict):
    # "<urn:uuid:6a8657b3-84d0-45df-b4b2-5fb6eef55ee5>" -> "6a8657b384d045dfb4b25fb6eef55ee5"
    # = compatible with duckdb UUID type
    return extract_uuid(sample["id"])


if __name__ == "__main__":
    langs = [
        "fry_Latn",
        "afr_Latn",
        "ita_Latn",
        "nld_Latn",
        "spa_Latn",
        "fra_Latn",
        "deu_Latn",
    ]

    def process_lang(lang: str):
        duckdb_path = f"/home/ampere/vanroy/CommonCrawl-CreativeCommons/duckdbs/fineweb-2/fw2-{lang}.duckdb"
        dataset_to_duckdb(
            "HuggingFaceFW/fineweb-2",
            duckdb_path,
            # For fineweb we still have to extract the UUID from the `id` column
            id_prep_func=fw2_prep_func,
            dataset_config=lang,
            overwrite=False,
            streaming=False,
            num_loaders=None,
            num_workers=64,
        )

    # Running all of them at the same time will require around 80GB of RAM but
    # will also put a strain on the file system.
    # max_workers = 3
    # with ProcessPoolExecutor(max_workers=min(max_workers, len(langs))) as p:
    #     p.map(process_lang, langs)

    for lang in tqdm(langs, unit="language", leave=True, position=0):
        process_lang(lang)
