import os
from pathlib import Path

import duckdb
from datasets import load_dataset
from huggingface_hub import HfApi

from commoncrawl_cc_annotation.utils import extract_uuid


hf_api = HfApi()


def duckdb_process(dumps: list[str], uuids: list[str], duckdb_path: str, added_key: str):
    batch = list(zip(dumps, uuids))
    con = duckdb.connect(duckdb_path, read_only=True)
    placeholders = ", ".join(["(?, ?)"] * len(batch))
    query = f"""
        SELECT CASE WHEN d.id IS NOT NULL THEN 1 ELSE 0 END AS exists
        FROM (VALUES {placeholders}) AS v(dump, id)
        LEFT JOIN data d
        ON v.dump = d.dump AND v.id = d.id;
    """

    results = con.execute(query, [value for pair in batch for value in pair]).fetchall()
    results = [bool(row[0]) for row in results]
    con.close()

    return {added_key: results}


def annotate_id_exists(
    dataset_name: str,
    duckdb_templ_path: str = "duckdbs/fw2-{lang}.duckdb",
    added_key: str = "id_found",
    num_loaders: int | None = None,
    num_workers: int | None = None,
    batch_size: int = 500,
    verbose: bool = False,
    dumps_to_process: list[str] | None = None,
    langs_to_process: list[str] | None = None,
):
    """
    Annotate a dataset with a key that indicates whether the ID is in the given DuckDB database.

    Args:
        dataset_name: The name of the dataset
        duckdb_templ_path: The path template to the DuckDB database. Must contain {lang} as a placeholder.
        added_key: The key to add to the dataset to indicate whether the ID is in the database
        num_loaders: The number of parallel processes to use for (down)loading the dataset
        num_workers: The number of parallel processes to use for annotation. Note that num_workers and batch_size impact memory usage!
        batch_size: The batch size for the annotation
        verbose: Whether to print the number of items in the dataset and the number of items in the database.
        dumps_to_process: The dumps to process. If None, all dumps are processed.
        langs_to_process: The languages to process. If None, all languages are processed.
    """
    dumps_to_process = dumps_to_process or []
    langs_to_process = langs_to_process or []

    # hf_api.snapshot_download(
    #     dataset_name,
    #     repo_type="dataset",
    #     cache_dir=None,
    #     local_dir=temp_dir,
    #     allow_patterns="*.parquet",
    #     max_workers=num_loaders,
    # )
    ptemp_dir = Path(temp_dir)
    print(ptemp_dir)
    for crawl_dir in ptemp_dir.joinpath("data").iterdir():
        if crawl_dir.is_dir():
            crawl = crawl_dir.stem
            if crawl not in dumps_to_process:
                continue
            for lang_dir in crawl_dir.iterdir():
                if lang_dir.is_dir():
                    lang = lang_dir.stem
                    if lang not in langs_to_process:
                        continue
                    duckdb_path = duckdb_templ_path.format(lang=lang)
                    data_dir = str(lang_dir).replace(temp_dir, "").lstrip("/")
                    ds = load_dataset(
                        "parquet", data_files=[str(f) for f in lang_dir.glob("*.parquet")], split="train"
                    )

                    ds = ds.map(
                        lambda cc_id: {"db_uuid": extract_uuid(cc_id)}, input_columns="id", num_proc=num_workers
                    )

                    ds = ds.map(
                        duckdb_process,
                        input_columns=["dump", "db_uuid"],
                        batch_size=batch_size,
                        num_proc=num_workers,
                        batched=True,
                        fn_kwargs={"duckdb_path": duckdb_path, "added_key": added_key},
                    )

                    ds = ds.remove_columns("db_uuid")

                    if verbose:
                        num_items = len(ds)
                        num_in_db = sum([1 for is_in_db in ds[added_key] if is_in_db])
                        print(
                            f"Dataset {dataset_name} ({added_key}): all={num_items:,}, in_db={num_in_db:,}, not_in_db={num_items - num_in_db:,}"
                        )

                    hf_api.delete_files(
                        dataset_name,
                        repo_type="dataset",
                        delete_patterns=[f"{data_dir}/*.parquet"],
                    )

                    ds.push_to_hub(
                        "BramVanroy/CommonCrawl-CreativeCommons",
                        config_name=f"{crawl}-{lang}",
                        data_dir=data_dir,
                    )
                    # 17.7


if __name__ == "__main__":
    dumps_to_process = ["CC-MAIN-2024-51"]
    langs_to_process = ["af"]
    langs_to_process = ["af", "nl", "fr", "de", "es", "it", "fy"]
    temp_dir = "/home/ampere/vanroy/CommonCrawl-CreativeCommons/tmp/CommonCrawl-CreativeCommons"
    duckdb_dir = "/home/ampere/vanroy/CommonCrawl-CreativeCommons/duckdbs/fineweb-2"

    annotate_id_exists(
        "BramVanroy/CommonCrawl-CreativeCommons",
        duckdb_templ_path=os.path.join(duckdb_dir, "fw2-{lang}.duckdb"),
        added_key="in_fw2",
        num_loaders=2,
        num_workers=2,
        batch_size=500,
        verbose=True,
        dumps_to_process=dumps_to_process,
        langs_to_process=langs_to_process,
    )
