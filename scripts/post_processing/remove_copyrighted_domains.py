import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import tldextract
from datasets import Features, load_dataset
from datasets.arrow_dataset import Dataset
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.hf_api import list_repo_files, upload_file

from commoncrawl_cc_annotation.script_utils import SCHEMA_NULLABLE


no_cache_extract = tldextract.TLDExtract(cache_dir=None)


def main(skip_dumps: list[str] = None, only_dumps: list[str] = None, num_proc: int | None = None):
    """
    Remove rows with domains that are known to be C&D'd from the CommonCrawl-CreativeCommons dataset.
    """
    skip_dumps = skip_dumps or []
    only_dumps = only_dumps or []

    ds_domains = load_dataset("BramVanroy/finewebs-copyright-domains", split="train")
    remove_domains = set(ds_domains.unique("domain"))
    del ds_domains
    print("Domains to remove:")
    print(remove_domains)

    ds_name = "BramVanroy/CommonCrawl-CreativeCommons"
    all_repo_files = list_repo_files(ds_name, repo_type="dataset")
    all_dumps = {f.split("/")[1] for f in all_repo_files if f.startswith("data/CC-MAIN")}

    for skip_dump in skip_dumps:
        if skip_dump not in all_dumps:
            raise ValueError(f"Dump {skip_dump} not found in the repository so cannot skip.")

    cfg_parquet_files = [f for f in all_repo_files if f.endswith(".parquet") and f.split("/")[1] not in skip_dumps]
    if only_dumps:
        cfg_parquet_files = [f for f in cfg_parquet_files if f.split("/")[1] in only_dumps]

    tmp_dir = str(Path(__file__).parent.parent / "tmp" / "remove_domains")
    for remote_parquet_uri in cfg_parquet_files:
        pf = Path(remote_parquet_uri)

        print(f"Processing {pf}")

        num_retries = 3
        while num_retries:
            try:
                local_fname = hf_hub_download(
                    ds_name,
                    filename=pf.name,
                    subfolder=pf.parent,
                    repo_type="dataset",
                    local_dir=tmp_dir,
                )
            except Exception as exc:
                num_retries -= 1
                print(f"Error downloading {pf}: {exc}")
                if not num_retries:
                    raise exc
            else:
                break

        features = Features.from_arrow_schema(SCHEMA_NULLABLE)
        ds = Dataset.from_parquet(local_fname, features=features)
        num_items = len(ds)

        def get_domain(url):
            extracted = no_cache_extract(url)
            return {"domain": f"{extracted.domain}.{extracted.suffix}"}

        ds = ds.map(get_domain, num_proc=num_proc, input_columns=["url"], desc="Extracting domains")
        ds = (
            ds.filter(
                lambda domain: domain not in remove_domains,
                input_columns=["domain"],
                desc="Filtering out removed domains",
                num_proc=num_proc,
            )
            .remove_columns("domain")
            .cast(features=features)
        )
        new_num_items = len(ds)

        if new_num_items < num_items:
            print(f"Removed {num_items - new_num_items} rows with removed domains in {pf}")
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
                        repo_id=ds_name,
                        commit_message="Remove rows with C&D domains",
                    )
                except Exception as exc:
                    num_retries -= 1
                    print(f"Error uploading {pf}: {exc}")
                    if not num_retries:
                        raise exc
                else:
                    break
        else:
            print(f"No rows with removed domains in {pf}")

        os.unlink(local_fname)

    shutil.rmtree(tmp_dir)
    print("Done")


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Remove rows with domains that are known to be C&D'd from the CommonCrawl-CreativeCommons dataset",
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
        "--num-proc",
        type=int,
        default=None,
        help="Number of processes to use for multiprocessing (default: None)",
    )

    cli_kwargs = vars(cparser.parse_args())
    main(**cli_kwargs)

"""
Done:
CC-MAIN-2019-30

TODO:
CC-MAIN-2020-05
CC-MAIN-2021-04
CC-MAIN-2022-05
CC-MAIN-2023-06
CC-MAIN-2024-51
CC-MAIN-2025-05

"""
