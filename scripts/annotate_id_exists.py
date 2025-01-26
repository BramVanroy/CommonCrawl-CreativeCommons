from typing import Callable

import duckdb
from datasets import load_dataset

from commoncrawl_cc_annotation.utils import extract_uuid


def duckdb_process(batch: list[str], duckdb_path: str, added_key: str, db_column_name: str):
    con = duckdb.connect(duckdb_path, read_only=True)
    results = {added_key: []}
    for uid in batch:
        uid = extract_uuid(uid)
        query = f"SELECT EXISTS (FROM data WHERE {db_column_name} = ?) AS id_exists;"
        result = con.execute(query, [uid]).fetchone()[0]
        results[added_key].append(result)
    con.close()
    return results


def annotate_id_exists(
    dataset_name: str,
    duckdb_path: str,
    unique_column_ds: str = "id",
    unique_column_db: str = "id",
    unique_column_func: Callable = None,
    dataset_config: str | None = None,
    push_to_hub: bool = False,
    added_key: str = "id_found",
    num_loaders: int | None = None,
    num_workers: int | None = None,
    batch_size: int = 1000,
    verbose: bool = False,
    split_name: str = "train",
    drop_columns: list[str] | None = None,
):
    """
    Annotate a dataset with a key that indicates whether the ID is in the given DuckDB database.

    Args:
        dataset_name: The name of the dataset
        duckdb_path: The path to the DuckDB database
        unique_column_ds: The name of the unique column in the dataset
        unique_column_db: The name of the unique column in the database
        unique_column_func: An optional function to apply to each sample to add the unique column
        dataset_config: The configuration of the dataset
        push_to_hub: Whether to push the annotated dataset to the Hub with the same name and config
        added_key: The key to add to the dataset to indicate whether the ID is in the database
        num_loaders: The number of parallel processes to use for (down)loading the dataset
        num_workers: The number of parallel processes to use for annotation. Note that num_workers and batch_size impact memory usage!
        batch_size: The batch size for the annotation
        verbose: Whether to print the number of items in the dataset and the number of items in the database.
        split_name: The split to annotate
        drop_columns: The columns to drop from the dataset after annotation
    """
    ds = load_dataset(dataset_name, dataset_config, split=split_name, num_proc=num_loaders)
    if unique_column_func is not None:
        ds = ds.map(unique_column_func, num_proc=num_workers)

    ds = ds.map(
        duckdb_process,
        input_columns=unique_column_ds,
        batch_size=batch_size,
        num_proc=num_workers,
        batched=True,
        fn_kwargs={"duckdb_path": duckdb_path, "added_key": added_key, "db_column_name": unique_column_db},
    )

    if drop_columns:
        ds = ds.remove_columns(drop_columns)

    if verbose:
        num_items = len(ds)
        num_in_db = sum([1 for is_in_db in ds[added_key] if is_in_db])
        print(
            f"Dataset {dataset_name} ({added_key}): all={num_items:,}, in_db={num_in_db:,}, not_in_db={num_items - num_in_db:,}"
        )

    if push_to_hub:
        ds.push_to_hub(dataset_name, config_name=dataset_config or "default")


def fw2_prep_func(sample: dict):
    # "<urn:uuid:6a8657b3-84d0-45df-b4b2-5fb6eef55ee5>" -> "6a8657b3-84d0-45df-b4b2-5fb6eef55ee5"
    # = compatible with duckdb UUID type
    return {"fw2_uuid": extract_uuid(sample["id"])}


def hplt2_prep_func(sample: dict):
    # They use a hash of the file path, URL, and timestamp, but this is currently not implemented here
    # because they seem to use a datestamp/ms timestamp, and not a string for the date
    # While that makes sense, I should do further testing to ensure the conversion
    # is done identical to theirs and can't readily find it in their codebase
    # https://github.com/hplt-project/monotextor-slurm/blob/629a45d0eae9528238072a086f71d978004cac4d/scripts/annotate.py#L179
    # return {"hplt2_uuid": xxh128_hexdigest(sample["file_path"] + sample["url"] + sample["date"])}
    # --------------------------------------------------------------------------------------^ should be converted to ms timestamp
    raise NotImplementedError


if __name__ == "__main__":
    # uid = "<urn:uuid:01e93f94-baa5-4666-a694-080ecb0b21c9>"
    # uid = extract_uuid(uid)
    # con = duckdb.connect("/home/ampere/vanroy/CommonCrawl-CreativeCommons/duckdbs/fw2-nld_Latn.duckdb")

    # # Get all "id" in database
    # ids = con.execute("SELECT id FROM data LIMIT 10;").fetchall()
    # # To python object
    # ids = [str(id[0]).replace("-", "") for id in ids]
    # print(ids)

    langs = [
        ("af", "afr_Latn"),
        ("nl", "nld_Latn"),
        ("fr", "fra_Latn"),
        ("de", "deu_Latn"),
        ("es", "spa_Latn"),
        ("it", "ita_Latn"),
        ("fy", "fry_Latn"),
    ]
    for dataset_name, subdir, prefix in [
        ("HuggingFaceFW/fineweb-2", "fineweb-2", "fw2"),
        ("HPLT/HPLT2.0_cleaned", "HPLT2.0_cleaned", "hplt2"),  # Won't work because of preprocessing
    ]:
        found_key = f"found_in_{prefix}"
        for shortlang, longlang in langs:
            duckdb_path = (
                f"/home/ampere/vanroy/CommonCrawl-CreativeCommons/duckdbs/{subdir}/{prefix}-{longlang}.duckdb"
            )
            try:
                annotate_id_exists(
                    "BramVanroy/CommonCrawl-CreativeCommons",
                    duckdb_path,
                    unique_column_func=fw2_prep_func if prefix == "fw2" else hplt2_prep_func,
                    unique_column_ds="fw2_uuid" if prefix == "fw2" else "id",
                    dataset_config=shortlang,
                    num_workers=96,
                    num_loaders=8,
                    added_key=found_key,
                    verbose=True,
                    push_to_hub=True,
                    drop_columns=["fw2_uuid"] if prefix == "fw2" else None,
                )
            except Exception as e:
                print(f"Error for {longlang} with {prefix}: {e}")
                continue
