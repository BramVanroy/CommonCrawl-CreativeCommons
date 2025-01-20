import duckdb
from datasets import load_dataset


def annotate_id_exists(
    dataset_name: str,
    duckdb_path: str,
    dataset_config: str | None = None,
    push_to_hub: bool = False,
    added_key: str = "id_found",
    num_proc: int | None = None,
    verbose: bool = False,
):
    """
    Annotate a dataset with a key that indicates whether the ID is in the given DuckDB database.

    Args:
        dataset_name: The name of the dataset
        duckdb_path: The path to the DuckDB database
        dataset_config: The configuration of the dataset
        push_to_hub: Whether to push the annotated dataset to the Hub with the same name and config
        added_key: The key to add to the dataset to indicate whether the ID is in the database
        num_proc: The number of parallel processes to use for annotation.
        verbose: Whether to print the number of items in the dataset and the number of items in the database.
    """
    ds = load_dataset(dataset_name, dataset_config)

    con = duckdb.connect(duckdb_path)

    def id_in_database(uid):
        uid = uid[10:-1]
        query = "SELECT EXISTS (FROM data WHERE id = ?) AS id_exists;"
        result = con.execute(query, [uid]).fetchone()[0]
        return {added_key: result}

    ds = ds.map(id_in_database, input_columns="id", num_proc=num_proc)

    if verbose:
        num_items = len(ds)
        num_in_db = sum([1 for is_i_db in ds[added_key] if is_i_db])
        print(f"Dataset {dataset_name}: all={num_items:,}, in_db={num_in_db:,}, not_in_db={num_items - num_in_db:,}")

    if push_to_hub:
        ds.push_to_hub(dataset_name, config_name=dataset_config or "default")
