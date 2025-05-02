"""THIS FILE IS DEPRECATED AND MIGHT NOT WORK. IT IS HERE FOR LEGACY REASONS FOR ME TO REMEMBER HOW I DID IT."""

import shutil
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from datasets.arrow_dataset import Dataset
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.hf_api import list_repo_files, upload_file

from commoncrawl_cc_annotation.data_utils import yield_repo_parquet_files
from commoncrawl_cc_annotation.script_utils import SCHEMA
from commoncrawl_cc_annotation.utils import extract_uuid


def check_eng_fw(ids, con):
    uuids = [extract_uuid(uid) for uid in ids]
    placeholders = ", ".join(["(?)"] * len(uuids))
    query = f"""
        SELECT CASE WHEN d.id IS NOT NULL THEN 1 ELSE 0 END AS exists
        FROM (VALUES {placeholders}) AS v(id)
        LEFT JOIN data d
        ON v.id = d.id;
    """
    results = con.execute(query, uuids).fetchall()

    results = [bool(r[0]) for r in results]

    return {"found_in_fw": results}


def check_fw2(ids, dump, con):
    uuids = [(dump, extract_uuid(uid)) for uid in ids]
    placeholders = ", ".join(["(?, ?)"] * len(uuids))
    query = f"""
        SELECT CASE WHEN d.id IS NOT NULL THEN 1 ELSE 0 END AS exists
        FROM (VALUES {placeholders}) AS v(dump, id)
        LEFT JOIN data d
        ON v.dump = d.dump AND v.id = d.id;
    """
    results = con.execute(query, [value for pair in uuids for value in pair]).fetchall()

    results = [bool(r[0]) for r in results]

    return {"found_in_fw": results}


def main(dump: str, fw_duckdb_tmpl: str, fw2_duckdb_tmpl: str, overwrite: bool = False):
    """
    1. Fix older versions of the dataset by:
    - Renaming the `found_in_fw2` column to `found_in_fw` if it exists
    - Adding the `found_in_fw` column if it does not exist
    - Filling the `found_in_fw` column with True/False values based on the containment in FineWeb(-2)
    2. Add `found_in_fw` column to the dataset if it does not exist and fill it with True/False
    values based on the containment in FineWeb(-2)

    Dumps that are too recent to be in FineWeb(-2) are set to None.

    Args:

    """
    ds_name = "BramVanroy/CommonCrawl-CreativeCommons"
    tmp_dir = Path(__file__).parents[2] / "tmp" / "add_containment"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = str(tmp_dir)

    all_repo_files = list_repo_files(ds_name, repo_type="dataset")
    all_dumps = {f.split("/")[1] for f in all_repo_files if f.startswith("data/CC-MAIN")}

    if dump not in all_dumps:
        raise ValueError(f"Dump {dump} not found in the repository.")

    dump_year = int(dump.split("-")[2])
    dump_issue = int(dump.split("-")[3])

    for remote_parquet_uri, local_fname in yield_repo_parquet_files(
        ds_name,
        tmp_dir=tmp_dir,
        only_dumps=[dump],
    ):
        pf = Path(remote_parquet_uri)
        lang = pf.parent.stem

        print(f"Processing {pf}")

        # This should error, in which we first try to back off to the FW2_SCHEMA, and if that does not work to NO_FW_SCHEMA
        table = pq.read_table(local_fname, schema=SCHEMA)

        changed_column_names = False
        if "found_in_fw2" in table.column_names:
            new_cols = ["found_in_fw" if col == "found_in_fw2" else col for col in table.column_names]
            table = table.rename_columns(new_cols)
            changed_column_names = True

        if "found_in_fw" in table.column_names and overwrite:
            print(f"Overwriting {pf}: dropping `found_in_fw` column")
            table = table.drop(["found_in_fw"])

        if "found_in_fw" not in table.column_names:
            print(f"Adding `found_in_fw` column to {pf} with null values")
            table = table.append_column(pa.field("found_in_fw", pa.bool_()), pc.chunked_array([None] * len(table)))
            changed_column_names = True

        if changed_column_names:
            pq.write_table(table, local_fname, compression="zstd")

        assert "found_in_fw" in table.column_names
        assert "found_in_fw2" not in table.column_names

        # Only do containment check if the dumps are not too recent that they are not in FW(2)
        # and skip containment if the column already exclusively contains True/False values
        needs_containment_fix = True
        if lang.startswith("eng"):
            # FW1 v1.3 contains data up to 2024-51
            if dump_year > 2024 or (dump_year == 2024 and dump_issue > 51):
                print(f"Skipping containment fix for {pf} because it is too recent for FW1")
                needs_containment_fix = False
        else:
            if dump_year > 2024 or (dump_year == 2024 and dump_issue > 18):
                print(f"Skipping containment fix for {pf} because it is too recent for FW2")
                needs_containment_fix = False

        uniq_values_found_in_fw = pc.unique(table["found_in_fw"]).to_pylist()
        if None not in uniq_values_found_in_fw:
            if not set(uniq_values_found_in_fw).issubset({True, False}):
                raise ValueError(
                    f"Error at {pf}: Unexpected values in the `found_in_fw` column. expected True/False: {uniq_values_found_in_fw}"
                )

            print(
                f"Skipping containment fix for {pf} because it already has only True/False values in the `found_in_fw` column"
            )
            needs_containment_fix = False

        # If we need to check containment, do so on the Dataset. Easier to work with.
        if needs_containment_fix:
            ds = Dataset.from_parquet(local_fname)

            if lang.startswith("eng"):
                pfw = Path(fw_duckdb_tmpl.format(dump=dump))
                duckdb_path = hf_hub_download(
                    repo_id="BramVanroy/fineweb-duckdbs", filename=pfw.name, local_dir=pfw.parent, repo_type="dataset"
                )
                con = duckdb.connect(duckdb_path, read_only=True)
                ds = ds.map(check_eng_fw, batched=True, fn_kwargs={"con": con}, input_columns="id", batch_size=10_000)
            else:
                pfw = Path(fw2_duckdb_tmpl.format(lang=lang))
                duckdb_path = hf_hub_download(
                    repo_id="BramVanroy/fineweb-2-duckdbs",
                    filename=pfw.name,
                    local_dir=pfw.parent,
                    repo_type="dataset",
                )
                con = duckdb.connect(duckdb_path, read_only=True)
                ds = ds.map(
                    check_fw2,
                    batched=True,
                    fn_kwargs={"con": con, "dump": dump},
                    input_columns="id",
                    batch_size=10_000,
                )

            con.close()
            ds.to_parquet(local_fname)

            assert "found_in_fw" in ds.column_names
            assert "found_in_fw2" not in ds.column_names

        # File was updated, either by only changing column names or by adding values to the `found_in_fw` column
        if changed_column_names or needs_containment_fix:
            # Test that the modified files can indeed be read correctly now with the right schema
            try:
                table = pq.read_table(local_fname, schema=SCHEMA)
            except Exception as exc:
                raise Exception(f"Could not read modified {local_fname}") from exc

            print(f"Uploading {pf} to {remote_parquet_uri}")
            num_retries = 3
            while num_retries:
                try:
                    upload_file(
                        path_or_fileobj=local_fname,
                        path_in_repo=remote_parquet_uri,
                        repo_type="dataset",
                        repo_id=ds_name,
                        commit_message="Fix/add found_in_fw column",
                    )
                except Exception as exc:
                    num_retries -= 1
                    print(f"Error uploading {pf}: {exc}")
                    if not num_retries:
                        raise exc
                else:
                    break

    shutil.rmtree(tmp_dir)
    print("Done")


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Fix the `found_in_fw` column in the CommonCrawl-CreativeCommons dataset from the hub. If an old `found_in_fw2` column is found, it is replaced."
        " If the `found_in_fw` column is missing, it is added. If the `found_in_fw` column contains None values, the containment check is run. Fixed files are reuploaded.",
    )
    cparser.add_argument("dump", type=str, help="Dump to process")
    cparser.add_argument(
        "--fw_duckdb_tmpl",
        type=str,
        default="duckdbs/fineweb/fw-{dump}.duckdb",
        help="Template for the FineWeb duckdb. Must contain the string '{dump}'",
    )
    cparser.add_argument(
        "--fw2_duckdb_tmpl",
        type=str,
        default="duckdbs/fineweb-2/fw2-{lang}_Latn.duckdb",
        help="Template for the FineWeb-2 duckdb. Must contain the string '{lang}'",
    )
    cparser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the `found_in_fw` column if it already exists",
    )

    cli_kwargs = vars(cparser.parse_args())
    main(**cli_kwargs)
