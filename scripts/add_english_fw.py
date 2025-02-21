from concurrent.futures import as_completed
from pathlib import Path
import shutil
from datasets.arrow_dataset import Dataset
import duckdb
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.hf_api import list_repo_files, upload_file
import pyarrow.parquet as pq

from datasets import  load_dataset

from commoncrawl_cc_annotation.utils import extract_uuid

TMP_DIR = "fix_eng_tmp"
LOCAL_TMPL_FW_DUCKDB = "duckdbs/fineweb/fw-{dump}.duckdb"
IGNORE_DUMPS = {}

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

def main():
    """
    Fix older versions of the dataset by:
    - renaming the column "found_in_fw2" to "found_in_fw"
    - adding actual values to the column "found_in_fw" to the English dataset by checking if it was in FW (not FW2)
    """
    ds_name = "BramVanroy/CommonCrawl-CreativeCommons"
    all_repo_files = list_repo_files(ds_name, repo_type="dataset")
    all_dumps = set([f.split("/")[1] for f in all_repo_files if f.startswith("data/CC-MAIN")])

    print(all_dumps)

    def has_fw2_column(cfg):
        ds = load_dataset(ds_name, cfg, streaming=True, split="train")
        try:
            feats = ds.features
            return "found_in_fw2" in feats
        except TypeError:
            return False

    cfgs_with_fw2 = [cfg for cfg in all_dumps if cfg not in IGNORE_DUMPS and has_fw2_column(cfg)]
    print(cfgs_with_fw2)

    futures = []
    for cfg in cfgs_with_fw2:
        cfg_parquet_files = [f for f in all_repo_files if f.endswith(".parquet") and f.startswith(f"data/{cfg}")]
        for remote_parquet_uri in cfg_parquet_files:
            pf = Path(remote_parquet_uri)
            print(f"Processing {pf}")

            local_fname = hf_hub_download(
                ds_name,
                filename=pf.name,
                subfolder=pf.parent,
                repo_type="dataset",
                local_dir=TMP_DIR,
            )
            table = pq.read_table(local_fname)

            if "found_in_fw2" not in table.column_names and "found_in_fw" in table.column_names:
                print(f"Skipping {pf}, already processed...")
                continue

            new_cols = ['found_in_fw' if col == 'found_in_fw2' else col for col in table.column_names]
            table = table.rename_columns(new_cols)
            pq.write_table(table, local_fname)

            is_eng = pf.parent.stem == "eng"

            if is_eng:
                pfw = Path(LOCAL_TMPL_FW_DUCKDB.format(dump=cfg))
                duckdb_path = hf_hub_download(repo_id="BramVanroy/fineweb-duckdbs", filename=pfw.name, local_dir=pfw.parent, repo_type="dataset")
                con = duckdb.connect(duckdb_path, read_only=True)
                ds = Dataset.from_parquet(local_fname)
                ds = ds.map(check_eng_fw, batched=True, fn_kwargs={"con": con}, input_columns="id", batch_size=1000)
                ds.to_parquet(local_fname)
                con.close()

            table = pq.read_table(local_fname)
            assert "found_in_fw" in table.column_names
            assert "found_in_fw2" not in table.column_names

            future = upload_file(
                path_or_fileobj=local_fname,
                path_in_repo=remote_parquet_uri,
                repo_type="dataset",
                repo_id=ds_name,
                commit_message="Fix found_in_fw column",
                run_as_future=True,
            )
            futures.append((future, local_fname))
    
    for future, local_fname in as_completed(futures):
        future.result()
        print(f"Uploaded {local_fname}")

    shutil.rmtree(TMP_DIR)
    print("Done")

if __name__ == "__main__":
    main()