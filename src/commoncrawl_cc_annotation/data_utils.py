from pathlib import Path
from typing import Generator

from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.hf_api import list_repo_files


def yield_repo_parquet_files(
    dataset_name: str,
    tmp_dir: str | None = None,
    only_dumps: list[str] | None = None,
    skip_dumps: list[str] | None = None,
    skip_files_with_suffix: list[str] | None = None,
    skip_non_fineweb_dumps: bool = False,
) -> Generator[tuple[str, str], None, None]:
    """
    Download all parquet files from the given repo, skipping the ones in skip_dumps
    and only keeping the ones in only_dumps. The files are downloaded to the tmp_dir.
    The files are downloaded one-by-one, and after each downloaded file its remote parquet URI
    as well as its local file path are yielded as a tuple.

    Args:
        repo_name (str): The name of the repo to download from.
        tmp_dir (str | None): The directory to download the files to. If None, a temporary directory is created.
        only_dumps (list[str] | None): A list of dumps to keep. If None, all dumps are kept.
        skip_dumps (list[str] | None): A list of dumps to skip. If None, no dumps are skipped.
        skip_files_with_suffix (list[str] | None): A list of suffixes to skip. If None, no suffixes are skipped.
        Useful if you want to skip specific files, e.g. data/CC-MAIN-2024-18/afr/000_00000.parquet.
        skip_non_fineweb_dumps (bool): If True, skip dumps that are not in FineWeb(-2).
    """
    skip_dumps = skip_dumps or []
    only_dumps = only_dumps or []
    skip_files_with_suffix = skip_files_with_suffix or []
    all_repo_files = list_repo_files(dataset_name, repo_type="dataset")
    all_dumps = {f.split("/")[1] for f in all_repo_files if f.startswith("data/CC-MAIN")}

    for skip_dump in skip_dumps:
        if skip_dump not in all_dumps:
            raise ValueError(f"Dump {skip_dump} not found in the repository so cannot skip.")

    cfg_parquet_files = [f for f in all_repo_files if f.endswith(".parquet") and f.split("/")[1] not in skip_dumps]
    if only_dumps:
        cfg_parquet_files = [f for f in cfg_parquet_files if f.split("/")[1] in only_dumps]

    tmp_dir = str(Path(__file__).parent.parent / "tmp" / "remove_domains")
    for remote_parquet_uri in cfg_parquet_files:
        if any(remote_parquet_uri.endswith(suffix) for suffix in skip_files_with_suffix):
            print(f"Skipping {remote_parquet_uri} because it ends with one of the suffixes...")
            continue

        pf = Path(remote_parquet_uri)
        dump_name = pf.parents[1].stem
        _, dump_year, dump_issue = dump_name.rsplit("-", 2)
        dump_year = int(dump_year)
        dump_issue = int(dump_issue)
        language = pf.parents[0].stem

        if skip_non_fineweb_dumps:
            do_process = True
            if (dump_year > 2024 or (dump_year == 2024 and dump_issue > 18)) and not language.startswith("eng"):
                do_process = False
                print(f"Skipping {pf} because it is not in English and the dump is after 2024-18")

            # FW1 v1.3 contains data up to 2024-51
            if (dump_year > 2024 or (dump_year == 2024 and dump_issue > 51)) and language.startswith("eng"):
                do_process = False
                print(f"Skipping {pf} because it is in English and the dump is after 2024-51")

            if not do_process:
                continue

        print(f"Processing {pf}")

        num_retries = 3
        while num_retries:
            try:
                local_fname = hf_hub_download(
                    dataset_name,
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
                yield remote_parquet_uri, local_fname
                break
