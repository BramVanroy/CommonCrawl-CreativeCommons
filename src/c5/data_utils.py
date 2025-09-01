import gzip
import io
import json
import time
from pathlib import Path
from typing import Generator

import requests
import yaml
from huggingface_hub import list_repo_files, upload_file
from huggingface_hub.file_download import hf_hub_download
from tqdm import tqdm

from c5.utils import is_in_fineweb


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
        language = pf.parents[0].stem

        if skip_non_fineweb_dumps:
            do_process = is_in_fineweb(dump_name, language)
            if not do_process:
                continue

        print(f"Processing {pf}")
        yield from download_and_yield_with_retry(
            dataset_name,
            remote_filename=pf.name,
            remote_subfolder=pf.parent,
            local_dir=tmp_dir,
        )


def download_and_yield_with_retry(
    repo_id: str,
    remote_filename: str,
    remote_subfolder: str | None = None,
    local_dir: str | None = None,
    num_retries: int = 3,
):
    while num_retries:
        try:
            local_fname = hf_hub_download(
                repo_id,
                filename=remote_filename,
                subfolder=remote_subfolder,
                repo_type="dataset",
                local_dir=local_dir,
            )
        except Exception as exc:
            num_retries -= 1
            print(f"Error downloading {remote_filename}: {exc}")
            if not num_retries:
                raise exc
        else:
            remote_uri = f"{remote_subfolder}/{remote_filename}" if remote_subfolder else remote_filename
            yield remote_uri, local_fname
            break


def upload_with_retry(
    repo_id: str,
    local_fname: str,
    remote_parquet_uri: str,
    num_retries: int = 3,
    run_as_future: bool = False,
):
    result = None
    while num_retries:
        try:
            result = upload_file(
                path_or_fileobj=local_fname,
                path_in_repo=remote_parquet_uri,
                repo_type="dataset",
                repo_id=repo_id,
                commit_message="Filtered on high-quality",
                run_as_future=run_as_future,
            )
        except Exception as exc:
            num_retries -= 1
            print(f"Error uploading {local_fname}: {exc}")
            if not num_retries:
                raise exc
        else:
            break

    return result


def upload_with_retry_async(
    repo_id: str,
    local_fname: str,
    remote_parquet_uri: str,
    num_retries: int = 3,
):
    return upload_with_retry(
        repo_id=repo_id,
        local_fname=local_fname,
        remote_parquet_uri=remote_parquet_uri,
        num_retries=num_retries,
        run_as_future=True,
    )


def get_fw2_language_threshold(languages: list[str]) -> dict[str, float]:
    """
    Get the language threshold for the given languages from the FineWeb-2 repository.

    Args:
        languages (list[str]): A list of languages to get the thresholds for.

    Returns:
        dict[str, float]: A dictionary mapping the languages to their thresholds.
    """
    pcfg_dir = Path(download_fw2_language_configs(languages))

    lang2threshold = {}
    for pfin in pcfg_dir.glob("*.yml"):
        lang = pfin.stem
        with pfin.open("r", encoding="utf-8") as fhin:
            cfg = yaml.safe_load(fhin)
        lang2threshold[lang] = cfg["language_score"]

    return lang2threshold


def download_fw2_language_configs(languages: list[str]) -> str:
    """
    Download the language configs from the FineWeb-2 repository.

    Args:
        languages (list[str]): A list of languages to download the configs for.

    Returns:
        str: The path to the directory where the configs are downloaded.
    """
    raw_url_template = "https://raw.githubusercontent.com/huggingface/fineweb-2/db2f99d2bde4a8ecde5552373876c9a1fad0bba3/configs/{language}.yml"
    pdout = Path(__file__).parents[2] / "language_configs"
    pdout.mkdir(parents=True, exist_ok=True)

    for language in languages:
        if language == "eng_Latn":
            continue

        pfout = pdout / f"{language}.yml"
        if pfout.exists():
            print(f"[{language}] Already downloaded, skipping...")
            continue

        download_url = raw_url_template.format(language=language)
        num_retries = 3
        wait_seconds = 10
        while num_retries > 0:
            try:
                file_resp = requests.get(download_url, timeout=10)
                file_resp.raise_for_status()
            except Exception as exc:
                num_retries -= 1
                if num_retries > 0:
                    print(f"[{language}] Retrying in {wait_seconds} seconds...")
                    time.sleep(wait_seconds)
                    wait_seconds *= 2
                else:
                    raise exc
            else:
                pfout.write_bytes(file_resp.content)
                break

    return str(pdout)


def yield_jsonl_gz_data_robust(pfiles: list[Path], disable_tqdm: bool = False):
    """
    Given a set of .jsonl.gz files, this function reads them in a robust way, skipping incomplete lines,
    and yielding one sample at a time (parse-able JSON line).

    :param pfiles: A list of .jsonl.gz files
    :return: A generator yielding the contents of the files
    """
    with tqdm(total=len(pfiles), desc="Reading", unit="file", disable=disable_tqdm) as pbar:
        for pfin in pfiles:
            if pfin.stat().st_size == 0:
                continue

            with gzip.open(pfin, "rt", encoding="utf-8") as fhin:
                num_failures = 0
                while True:
                    try:
                        line = fhin.readline()
                        if not line:
                            break  # End of currently available content
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        # Handle partial or malformed JSON (incomplete writes)
                        num_failures += 1
                    except EOFError:
                        # Handle unexpected EOF in gzip
                        num_failures += 1
                        break
                if num_failures:
                    print(f"Skipped {num_failures:,} corrupt line(s) in {pfin}")
            pbar.update(1)


def download_warc_urls_file(dump: str, output_folder: str, limit: int = None, overwrite: bool = False) -> str:
    pdout = Path(output_folder)
    pdout.mkdir(parents=True, exist_ok=True)
    pfout = pdout / f"{dump}_warc_urls.txt"

    if (pfout.exists() and pfout.stat().st_size > 0) and not overwrite:
        print(f"File {pfout} already exists. Skipping download.")
        return str(pfout)
    elif pfout.exists() and overwrite:
        print(f"File {pfout} already exists. Overwriting.")
        pfout.unlink()

    # A file with one path per line, e.g.
    paths_url = f"https://data.commoncrawl.org/crawl-data/{dump}/warc.paths.gz"

    r = requests.get(paths_url, timeout=120)
    r.raise_for_status()

    lines = gzip.GzipFile(fileobj=io.BytesIO(r.content)).read().decode("utf-8").splitlines()

    # Keep only WARC files (you can also sample/limit here)
    warc_urls = [p for p in lines if p.endswith(".warc.gz")]

    if limit is not None:
        warc_urls = warc_urls[:limit]

    with pfout.open("w", encoding="utf-8") as fhout:
        fhout.write("\n".join(warc_urls) + "\n")

    return str(pfout)
