import re

import pyarrow as pa
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import (
    LanguageFilter,
    URLFilter,
)
from datatrove.pipeline.formatters import FTFYFormatter, PIIFormatter, SymbolLinesFormatter
from datatrove.pipeline.readers import JsonlReader, WarcReader
from datatrove.pipeline.writers import HuggingFaceDatasetWriter, JsonlWriter
from pydantic import BaseModel

from .components.annotators import DatabaseContainmentAnnotator, LicenseAnnotator
from .components.filters import EmptyTextFilter, LicenseFilter


# Afrikaans, German, English, French, Frisian, Italian, Dutch, Spanish
LANGUAGES = [
    "afr_Latn",
    "deu_Latn",
    "eng_Latn",
    "fra_Latn",
    "fry_Latn",
    "ita_Latn",
    "nld_Latn",
    "spa_Latn",
]


class _BaseConfig(BaseModel):
    """Base Config class for local and Slurm configurations"""

    tasks: int = 1
    randomize_start_duration: int = 0
    language_threshold: float = 0.65
    languages: list = LANGUAGES
    keep_with_license_only: bool = True
    duckdb_templ_path: str | None = None
    ignore_duckdb_for: list[str] = ["eng_Latn"]
    limit: int = -1


class LocalConfig(_BaseConfig):
    """Local configuration for running the pipeline on a local machine"""

    workers: int = -1


class SlurmConfig(_BaseConfig):
    """Slurm configuration for running the pipeline on a cluster"""

    time: str
    mem_per_cpu_gb: int = 2
    cpus_per_task: int = 1


def build_pipeline(
    dump: str,
    output_path: str,
    duckdb_templ_path: str,
    ignore_duckdb_for: list[str],
    languages: list[str],
    language_threshold: float = 0.65,
    limit: int = -1,
) -> list[PipelineStep]:
    """Build a pipeline for extracting and filtering web pages from Common Crawl. This is a separate
    function so that it can be used in both the local and Slurm scripts.

    Args:
        dump (str): Common Crawl dump to process
        output_path (str): Main output path. JSONL.GZ files will be saved in subfolders based on language
        duckdb_templ_path (str): Path to the DuckDB databases. Must contain the placeholder '{lang}'.
        Example: "data/duckdbs/fw2-{lang}.db". `lang` will be replaced with the language code
        as found in the concatenation of `{metadata["language"]}_{metadata["language_script"]}`.
        languages (list[str]): List of languages to filter for
        language_threshold (float, optional): Minimum language detection threshold. Defaults to 0.65.
        limit (int, optional): Maximum number of pages to process per task, useful for debugging.
        -1 = no limit. Defaults to -1.

    Returns:
        list[PipelineStep]: List of pipeline steps (i.e., the pipeline components)
    """
    if not duckdb_templ_path or "{language}" not in duckdb_templ_path:
        raise ValueError("The duckdb_templ_path must contain the placeholder '{language}'")

    return [
        WarcReader(
            f"s3://commoncrawl/crawl-data/{dump}/segments/",
            glob_pattern="*/warc/*",
            default_metadata={"dump": dump},
            limit=limit,
        ),
        URLFilter(),
        EmptyTextFilter(),  # filter items with empty HTML (text = read HTML at this point)
        LicenseAnnotator(),
        LicenseFilter(),
        Trafilatura(favour_precision=True, timeout=600.0),
        EmptyTextFilter(),  # filter items with empty extracted text
        LanguageFilter(backend="glotlid", languages=languages, language_threshold=language_threshold),
        DatabaseContainmentAnnotator(
            duckdb_templ_path=duckdb_templ_path,
            ignore_duckdb_for=ignore_duckdb_for,
            added_key="found_in_fw2",
        ),
        # From FW2: https://github.com/huggingface/fineweb-2/blob/main/fineweb-2-pipeline.py
        FTFYFormatter(),  # fix encoding issues. Important in a multilingual setting
        PIIFormatter(),  # remove PII
        SymbolLinesFormatter(symbols_to_remove=["|"], replace_char="\n"),  # fix trafilatura table artifacts
        JsonlWriter(
            output_folder=f"{output_path}/",
            output_filename="${language}/${rank}.jsonl.gz",
            expand_metadata=True,
        ),
    ]


SCHEMA = pa.schema(
    [
        ("text", pa.string()),
        ("id", pa.string()),
        ("dump", pa.string()),
        ("url", pa.string()),
        ("date", pa.string()),
        ("file_path", pa.string()),
        ("license_abbr", pa.string()),
        ("license_version", pa.string()),
        ("license_location", pa.string()),
        ("license_in_head", pa.bool_()),
        ("license_in_footer", pa.bool_()),
        (
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
        ("license_parse_error", pa.bool_()),
        ("license_disagreement", pa.bool_()),
        ("language", pa.string()),
        ("language_script", pa.string()),
        ("language_score", pa.float64()),
        ("found_in_fw2", pa.bool_()),
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
            output_filename="data/${dump}/${language}/${rank}.parquet",
            compression="zstd",
            schema=SCHEMA,
            cleanup=True,
        ),
    ]
