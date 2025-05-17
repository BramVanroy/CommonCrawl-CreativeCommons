from pathlib import Path
from typing import Generator
import yaml
import requests
import time
from pathlib import Path

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

def get_fw2_language_threshold(languages: list[str] | None = None) -> dict[str, float]:
    pcfg_dir = Path(download_fw2_language_configs(languages))

    lang2threshold = {}
    for pfin in pcfg_dir.glob("*.yml"):
        lang = pfin.stem
        with pfin.open("r", encoding="utf-8") as fhin:
            cfg = yaml.safe_load(fhin)
        lang2threshold[lang] = cfg["language_score"]

    return lang2threshold
                   

def download_fw2_language_configs(languages: list[str] | None = None) -> str:
    """
    Downloads all .yaml or .yml files from the GitHub repository's configs directory
    (https://github.com/huggingface/fineweb-2/tree/main/configs) into the specified
    local output directory 'pdout'. A simple retry mechanism is included.

    Args:
        languages (list[str] | None): A list of languages to download. If None, all languages are downloaded.

    Returns:
        str: The path to the directory where the files were downloaded.
    """
    
    repo_api_url = "https://api.github.com/repos/huggingface/fineweb-2/contents/configs"

    pdout = Path(__file__).parents[2] / "language_configs"
    pdout.mkdir(parents=True, exist_ok=True)

    num_retries = 3
    wait_seconds = 10
    while num_retries > 0:
        try:
            # An API overview of all files in the configs directory
            response = requests.get(repo_api_url)
            response.raise_for_status()
        except Exception as exc:
            num_retries -= 1
            if num_retries > 0:
                print(f"Retrying in {wait_seconds} seconds...")
                time.sleep(wait_seconds)
                wait_seconds *= 2
            else:
                raise exc
        else:
            files_info = response.json()
            break

    for item in files_info:
        if item["type"] == "file" and item["name"].endswith(".yml"):
            pfout = pdout / item["name"]

            if pfout.exists() and pfout.stat().st_size > 0:
                print(f"Skipping {pfout} because it already exists and is not empty.")
                continue

            if languages is not None and pfout.stem not in languages:
                print(f"Skipping {pfout} because it is not in the specified languages.")
                continue
            
            num_retries = 3
            wait_seconds = 10
            while num_retries > 0:
                try:
                    file_resp = requests.get(item["download_url"], timeout=10)
                    file_resp.raise_for_status()  
                except Exception as exc:
                    num_retries -= 1
                    if num_retries > 0:
                        print(f"Retrying in {wait_seconds} seconds...")
                        time.sleep(wait_seconds)
                        wait_seconds *= 2
                    else:
                        raise exc
                else:             
                    pfout.write_bytes(file_resp.content)
                    break 

    return str(pdout)