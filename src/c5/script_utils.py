import re
from pathlib import Path
from typing import Literal

import pyarrow as pa
from datasets import load_dataset
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import URLFilter
from datatrove.pipeline.formatters import FTFYFormatter, PIIFormatter, SymbolLinesFormatter
from datatrove.pipeline.readers import JsonlReader, WarcReader
from datatrove.pipeline.writers import HuggingFaceDatasetWriter, JsonlWriter
from huggingface_hub import hf_hub_download, list_repo_files
from pydantic import BaseModel

from c5.components.annotators import FWDBContainmentAnnotator, FWSingleDBContainmentAnnotator, LicenseAnnotator
from c5.components.filters import EmptyTextFilter, LanguageFilterWithIgnore, LicenseFilter
from c5.data_utils import get_fw2_language_threshold


LANGUAGES_V1 = [
    "afr_Latn",
    "deu_Latn",
    "eng_Latn",
    "fra_Latn",
    "fry_Latn",
    "ita_Latn",
    "nld_Latn",
    "spa_Latn",
]
LANGUAGES_EU = [
    "bul_Cyrl",
    "hrv_Latn",
    "ces_Latn",
    "dan_Latn",
    "nld_Latn",
    "eng_Latn",
    "ekk_Latn",
    "fin_Latn",
    "fra_Latn",
    "deu_Latn",
    "ell_Grek",
    "hun_Latn",
    "gle_Latn",
    "ita_Latn",
    "lvs_Latn",
    "lit_Latn",
    "mlt_Latn",
    "pol_Latn",
    "por_Latn",
    "ron_Latn",
    "slk_Latn",
    "slv_Latn",
    "spa_Latn",
    "swe_Latn"
]


class BaseUploadConfig(BaseModel):
    """Base Config class for local and Slurm configurations"""

    tasks: int = 1
    workers: int = -1
    randomize_start_duration: int = 0
    limit: int = -1


class SlurmUploadConfig(BaseUploadConfig):
    """Slurm configuration for running the pipeline on a cluster"""

    time: str
    mem_per_cpu_gb: int = 2
    cpus_per_task: int = 1


class BaseConfig(BaseModel):
    """Base Config class for local and Slurm configurations"""

    main_tasks: int = 1
    containment_tasks: int = 1
    main_workers: int = -1
    containment_workers: int = -1

    randomize_start_duration: int = 0
    languages: list | None | Literal["v1", "eu"] = None
    ignore_undetermined: bool = True
    fw_duckdb_templ_path: str | None = None
    fw2_duckdb_templ_path: str | None = None
    overwrite_with_none: bool = False
    ignore_duckdb_for: list[str] | None = None
    limit: int = -1

    def model_post_init(self, __context):
        if not self.languages:
            self.languages = retrieve_supported_languages()
        elif self.languages == "v1":
            self.languages = LANGUAGES_V1
        elif self.languages == "eu":
            self.languages = LANGUAGES_EU


class SlurmConfig(BaseConfig):
    """Slurm configuration for running the pipeline on a cluster"""

    main_time: str = "3-00:00:00"
    containment_time: str = "1-00:00:00"
    main_mem_per_cpu_gb: int = 4
    containment_mem_per_cpu_gb: int = 4
    main_cpus_per_task: int = 1
    containment_cpus_per_task: int = 16
    max_array_launch_parallel: bool = False
    stagger_max_array_jobs: int = 900


def build_main_pipeline(
    dump: str,
    output_folder: str,
    languages: list[str] | None,
    limit: int = -1,
    extra_domains: list[str] | None = None,
    ignore_undetermined: bool = True,
) -> list[PipelineStep]:
    """Build a pipeline for extracting and filtering web pages from Common Crawl. This is a separate
    function so that it can be used in both the local and Slurm scripts.

    Args:
        dump (str): Common Crawl dump to process
        output_folder (str): Path to the output directory. Files will be written as
        ${language}_${language_script}/${rank}.jsonl.gz
        languages (list[str]): List of languages to filter for or None
        limit (int, optional): Maximum number of pages to process per task, useful for debugging.
        -1 = no limit. Defaults to -1.
        extra_domains (list[str], optional): List of extra domains to filter for. Defaults to None.

    Returns:
        list[PipelineStep]: List of pipeline steps (i.e., the pipeline components)
    """
    # Do not include any of GlotLID's nonlinguistic and undetermined languages: https://github.com/cisnlp/GlotLID/blob/main/languages-v3.md
    ignore_languages = ["und", "zxx"] if ignore_undetermined else []
    fw2_languages = [l for l in languages if l != "eng_Latn"]
    lang_thresholds = get_fw2_language_threshold(fw2_languages)

    if "eng_Latn" in languages:
        # Add the English language threshold to the thresholds
        lang_thresholds["eng_Latn"] = 0.65
    
    if languages is not None:
        for lang in languages:
            if lang not in lang_thresholds:
                raise ValueError(f"Language {lang} not found in the language thresholds. Something must have gone wrong when loading the data.")

    return [
        WarcReader(
            f"s3://commoncrawl/crawl-data/{dump}/segments/",
            glob_pattern="*/warc/*",
            default_metadata={"dump": dump},
            limit=limit,
        ),
        URLFilter(extra_domains=extra_domains),
        EmptyTextFilter(),  # filter items with empty HTML (text-attr = read HTML at this point) -- cheap
        LicenseAnnotator(),
        LicenseFilter(),
        # Setting deduplicate to False because of its destructive nature -- https://github.com/adbar/trafilatura/issues/778
        Trafilatura(favour_precision=True, timeout=60.0, deduplicate=False),
        EmptyTextFilter(),  # filter items with empty extracted text -- should be rare but it's cheap
        LanguageFilterWithIgnore(
            languages=languages,
            ignore_language_prefixes=ignore_languages,
            language_threshold=lang_thresholds
        ),
        # From FW2: https://github.com/huggingface/fineweb-2/blob/main/fineweb-2-pipeline.py:
        FTFYFormatter(),  # fix encoding issues. Important in a multilingual setting
        PIIFormatter(),  # remove PII
        SymbolLinesFormatter(symbols_to_remove=["|"], replace_char="\n"),  # fix trafilatura table artifacts
        JsonlWriter(
            output_folder=f"{output_folder}/",
            output_filename="${language}_${language_script}/${rank}.jsonl.gz",
        ),
    ]


def build_containment_pipeline(
    input_path: str,
    duckdb_path: str,
    is_fw2: bool,
    output_folder: str,
    overwrite_with_none: bool = False,
) -> list[PipelineStep]:
    """
    Build a pipeline for annotating the web pages with the database containment information.

    Args:
        input_path (str): Path to the input directory.
        duckdb_path (str): Path to the FineWeb DuckDB database (English) (filled in template URI).
        is_fw2 (bool): Whether the database is FineWeb-2 or not.
        output_path (str): Path to use for the output directory. Files will be written as
        ${language}_${language_script}/${rank}.jsonl.gz
        overwrite_with_none (bool, optional): If True, the 'found' field will be set to `None` for all documents,
        regardless of the language. Useful if you know that a given dump does not occur in the other database.
        This improves speed as the database is not queried. Defaults to False.
    """
    return [
        JsonlReader(
            data_folder=input_path,
            glob_pattern="**/*.jsonl.gz",
        ),
        FWSingleDBContainmentAnnotator(
            duckdb_path=duckdb_path,
            is_fw2=is_fw2,
            added_key="found_in_fw",
            overwrite_with_none=overwrite_with_none,
        ),
        JsonlWriter(
            output_folder=f"{output_folder}/",
            output_filename="${language}_${language_script}/${rank}.jsonl.gz",
            expand_metadata=True,
        ),
    ]


def get_fw_c_and_d_domains() -> set[str]:
    """
    Get the domains that were removed from FineWeb(-2) as a result from a cease-and-desist letter,
    collecting in this repository: BramVanroy/finewebs-copyright-domains

    These domains can then be forwarded to the URL filter's `extra_domains`, ensuring that
    these domains are not included in our dataset.
    """
    ds = load_dataset("BramVanroy/finewebs-copyright-domains", split="train")
    return set(ds["domain"])


SCHEMA = pa.schema(
    [
        pa.field("text", pa.string(), nullable=False),
        pa.field("id", pa.string(), nullable=False),
        pa.field("dump", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=False),
        pa.field("date", pa.string(), nullable=False),
        pa.field("file_path", pa.string(), nullable=False),
        pa.field("license_abbr", pa.string(), nullable=False),
        pa.field("license_version", pa.string(), nullable=True),
        pa.field("license_location", pa.string(), nullable=False),
        pa.field("license_in_head", pa.bool_(), nullable=False),
        pa.field("license_in_footer", pa.bool_(), nullable=False),
        pa.field(
            "potential_licenses",
            pa.struct(
                [
                    pa.field("abbr", pa.list_(pa.string()), nullable=False),
                    pa.field("in_footer", pa.list_(pa.bool_()), nullable=False),
                    pa.field("in_head", pa.list_(pa.bool_()), nullable=False),
                    pa.field("location", pa.list_(pa.string()), nullable=False),
                    pa.field("version", pa.list_(pa.string()), nullable=False),
                ]
            ),
        ),
        pa.field("license_parse_error", pa.bool_(), nullable=False),
        pa.field("license_disagreement", pa.bool_(), nullable=False),
        pa.field("language_script", pa.string(), nullable=False),
        pa.field("language", pa.string(), nullable=False),
        pa.field("language_score", pa.float64(), nullable=False),
        pa.field("found_in_fw", pa.bool_(), nullable=True),
    ]
)

SCHEMA_NULLABLE = pa.schema(
    [
        pa.field("text", pa.string()),
        pa.field("id", pa.string()),
        pa.field("dump", pa.string()),
        pa.field("url", pa.string()),
        pa.field("date", pa.string()),
        pa.field("file_path", pa.string()),
        pa.field("license_abbr", pa.string()),
        pa.field("license_version", pa.string()),
        pa.field("license_location", pa.string()),
        pa.field("license_in_head", pa.bool_()),
        pa.field("license_in_footer", pa.bool_()),
        pa.field(
            "potential_licenses",
            pa.struct(
                [
                    pa.field("abbr", pa.list_(pa.string())),
                    pa.field("in_footer", pa.list_(pa.bool_())),
                    pa.field("in_head", pa.list_(pa.bool_())),
                    pa.field("location", pa.list_(pa.string())),
                    pa.field("version", pa.list_(pa.string())),
                ]
            ),
        ),
        pa.field("license_parse_error", pa.bool_()),
        pa.field("license_disagreement", pa.bool_()),
        pa.field("language_script", pa.string()),
        pa.field("language", pa.string()),
        pa.field("language_score", pa.float64()),
        pa.field("found_in_fw", pa.bool_()),
    ]
)

FW2_SCHEMA = pa.schema(
    [
        pa.field("text", pa.string(), nullable=False),
        pa.field("id", pa.string(), nullable=False),
        pa.field("dump", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=False),
        pa.field("date", pa.string(), nullable=False),
        pa.field("file_path", pa.string(), nullable=False),
        pa.field("license_abbr", pa.string(), nullable=False),
        pa.field("license_version", pa.string(), nullable=True),
        pa.field("license_location", pa.string(), nullable=False),
        pa.field("license_in_head", pa.bool_(), nullable=False),
        pa.field("license_in_footer", pa.bool_(), nullable=False),
        pa.field(
            "potential_licenses",
            pa.struct(
                [
                    pa.field("abbr", pa.list_(pa.string()), nullable=False),
                    pa.field("in_footer", pa.list_(pa.bool_()), nullable=False),
                    pa.field("in_head", pa.list_(pa.bool_()), nullable=False),
                    pa.field("location", pa.list_(pa.string()), nullable=False),
                    pa.field("version", pa.list_(pa.string()), nullable=False),
                ]
            ),
        ),
        pa.field("license_parse_error", pa.bool_(), nullable=False),
        pa.field("license_disagreement", pa.bool_(), nullable=False),
        pa.field("language_script", pa.string(), nullable=False),
        pa.field("language", pa.string(), nullable=False),
        pa.field("language_score", pa.float64(), nullable=False),
        pa.field("found_in_fw2", pa.bool_(), nullable=True),
    ]
)

NO_FW_SCHEMA = pa.schema(
    [
        pa.field("text", pa.string(), nullable=False),
        pa.field("id", pa.string(), nullable=False),
        pa.field("dump", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=False),
        pa.field("date", pa.string(), nullable=False),
        pa.field("file_path", pa.string(), nullable=False),
        pa.field("license_abbr", pa.string(), nullable=False),
        pa.field("license_version", pa.string(), nullable=True),
        pa.field("license_location", pa.string(), nullable=False),
        pa.field("license_in_head", pa.bool_(), nullable=False),
        pa.field("license_in_footer", pa.bool_(), nullable=False),
        pa.field(
            "potential_licenses",
            pa.struct(
                [
                    pa.field("abbr", pa.list_(pa.string()), nullable=False),
                    pa.field("in_footer", pa.list_(pa.bool_()), nullable=False),
                    pa.field("in_head", pa.list_(pa.bool_()), nullable=False),
                    pa.field("location", pa.list_(pa.string()), nullable=False),
                    pa.field("version", pa.list_(pa.string()), nullable=False),
                ]
            ),
        ),
        pa.field("license_parse_error", pa.bool_(), nullable=False),
        pa.field("license_disagreement", pa.bool_(), nullable=False),
        pa.field("language_script", pa.string(), nullable=False),
        pa.field("language", pa.string(), nullable=False),
        pa.field("language_score", pa.float64(), nullable=False),
    ]
)


def job_id_retriever(job_id: str) -> str:
    return re.search(r"Submitted batch job (\d+)", job_id).group(1)


def build_upload_pipeline(jsonl_path: str, output_path: str, hf_repo: str, limit: int = -1) -> list[PipelineStep]:
    return [
        JsonlReader(
            jsonl_path,
            limit=limit,
            glob_pattern="**/*.jsonl.gz",
        ),
        HuggingFaceDatasetWriter(
            hf_repo,
            local_working_dir=output_path,
            private=True,
            output_filename="data/${dump}/${language}_${language_script}/${rank}.parquet",
            compression="zstd",
            schema=SCHEMA,
            cleanup=True,
        ),
    ]


def get_dumps_with_duckdb(
    dump_name: str, languages: list[str], ignore_duckdb_for: list[str] = None
) -> tuple[list[str], bool]:
    """
    Get the list of languages that should be ignored when querying the DuckDB databases.
    This is useful for dumps that are too recent and do not have the data in the DuckDB databases yet.
    The function also returns a boolean indicating whether all languages should be ignored.

    Args:
        dump_name (str): Name of the dump (e.g., "CC-MAIN-2024-18")
        ignore_duckdb_for (list[str]): Initial list of languages to ignore when querying the DuckDB databases,
        for instance provided by the user in the config.
        languages (list[str]): List of languages to process.

    Returns:
        tuple[list[str], bool]: A tuple containing:
            - List of languages to ignore when querying the DuckDB databases, e.g. ["eng_Latn", "fra_Latn"].
            - Boolean indicating whether all languages should be ignored.
    """
    languages = languages or []
    dump_year = int(dump_name.split("-")[2])
    dump_issue = int(dump_name.split("-")[3])

    # Not in FineWeb-2
    ignore_duckdb_for = ignore_duckdb_for or []

    if dump_year > 2024 or (dump_year == 2024 and dump_issue > 18):
        ignore_duckdb_for += [lang for lang in languages if lang not in ["eng_Latn", "eng", "en"]]

    # FW1 v1.3 contains data up to 2024-51
    if dump_year > 2024 or (dump_year == 2024 and dump_issue > 51):
        ignore_duckdb_for += ["eng_Latn"]

    ignore_all_duckdb = set(ignore_duckdb_for) == set(languages)

    return ignore_duckdb_for, ignore_all_duckdb

def retrieve_supported_languages(include_english: bool = True) -> list[str]:
    """
    Retrieve the list of supported languages from the FineWeb-2 DuckDB databases.
    This function downloads the list of files from the Hugging Face repository and extracts
    the language codes from the filenames. The function also adds "eng_Latn" to the list of languages.

    Args:
        include_english (bool): Whether to include "eng_Latn" in the list of languages. Defaults to True.

    Returns:
        list[str]: List of supported languages, e.g. ["eng_Latn", "fra_Latn", "aak_Latn", ...].
    """
    fnames = list_repo_files(repo_id="BramVanroy/fineweb-2-duckdbs", repo_type="dataset",)
    languages = [f.replace("fw2-", "").replace(".duckdb", "") for f in fnames if f.endswith(".duckdb") and "_removed" not in f]
    if include_english:
        languages += ["eng_Latn"]
    return languages

def download_duckdbs(dump_name: str, fw_duckdb_path: str, cfg: BaseConfig) -> tuple[list[str], bool]:
    """
    Download the DuckDB databases from Hugging Face if they are not already present on-disk, and verify
    which languages should be ignored when querying the databases and whether ALL should be ignored.
    This is useful because it would allow us to set the 'found' field to None for all documents.
    See FWDatabaseContainmentAnnotator for more details.

    WARNING: if you're running the pipeline on all languages, this will have to download
    all fineweb-2 duckdbs immediately. This will amount to around ~300GB of data! If you
    do not want that, either be specific in your config about the languages you want to process,
    or do not run the containment pipeline.

    Args:
        dump_name (str): Name of the dump (e.g., "CC-MAIN-2024-18")
        fw_duckdb_path (str): Path to the FineWeb DuckDB database (English) (filled in template URI).
        cfg (BaseConfig): Configuration object containing the paths to the databases and other settings.

    Returns:
        tuple[list[str], bool]: A tuple containing:
            - List of languages to ignore when querying the DuckDB databases, e.g. ["eng_Latn", "fra_Latn"].
            - Boolean indicating whether all languages should be ignored.
    """
    if "{language}" not in cfg.fw2_duckdb_templ_path:
        raise ValueError("The fw2_duckdb_templ_path must contain the string '{language}'")

    if "{dump}" not in cfg.fw_duckdb_templ_path:
        raise ValueError("The fw_duckdb_templ_path must contain the string '{dump}'")

    ignore_duckdb_for, ignore_all_duckdb = get_dumps_with_duckdb(
        dump_name,
        cfg.languages,
        cfg.ignore_duckdb_for,
    )

    # Only download languages that are not in ignore_duckdb_for (occurs when crawl is too recent or in config ignore_duckdb_for)
    duckdb_languages = []
    for lang in cfg.languages:
        if lang not in ignore_duckdb_for:
            duckdb_languages.append(lang)

    if duckdb_languages:
        pfw = Path(fw_duckdb_path)
        for lang in duckdb_languages:
            if lang == "eng_Latn":
                if not pfw.exists() or pfw.stat().st_size == 0:
                    hf_hub_download(
                        repo_id="BramVanroy/fineweb-duckdbs",
                        filename=pfw.name,
                        local_dir=pfw.parent,
                        repo_type="dataset",
                    )
            else:
                local_path = cfg.fw2_duckdb_templ_path.format(language=lang)
                pf = Path(local_path)
                if not pf.exists() or pf.stat().st_size == 0:
                    hf_hub_download(
                        repo_id="BramVanroy/fineweb-2-duckdbs",
                        filename=pf.name,
                        local_dir=pf.parent,
                        repo_type="dataset",
                    )

    return ignore_duckdb_for, ignore_all_duckdb
