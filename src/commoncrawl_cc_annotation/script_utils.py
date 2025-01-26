import re

import pyarrow as pa
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import (
    LanguageFilter,
    URLFilter,
)
from datatrove.pipeline.readers import JsonlReader, WarcReader
from datatrove.pipeline.writers import HuggingFaceDatasetWriter, JsonlWriter
from pydantic import BaseModel

from .components.annotators import HtmlCopier, LicenseAnnotator
from .components.filters import EmptyTextFilter, LicenseFilter


# Dutch, Frisian, English, Spanish, French, Italian, German, Afrikaans
LANGUAGES = [
    "nl",
    "fy",
    "en",
    "es",
    "fr",
    "it",
    "de",
    "af",
]


class _BaseConfig(BaseModel):
    """Base Config class for local and Slurm configurations"""

    tasks: int = 1
    randomize_start_duration: int = 0
    language_threshold: float = 0.65
    languages: list = LANGUAGES
    keep_with_license_only: bool = True
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
    languages: list[str],
    language_threshold: float = 0.65,
    limit: int = -1,
    trafiltura_first: bool = False,
) -> list[PipelineStep]:
    """Build a pipeline for extracting and filtering web pages from Common Crawl. This is a separate
    function so that it can be used in both the local and Slurm scripts.

    Args:
        dump (str): Common Crawl dump to process
        output_path (str): Main output path. JSONL.GZ files will be saved in subfolders based on language
        languages (list[str]): List of languages to filter for
        language_threshold (float, optional): Minimum language detection threshold. Defaults to 0.65.
        limit (int, optional): Maximum number of pages to process per task, useful for debugging.
        -1 = no limit. Defaults to -1.
        trafiltura_first (bool, optional): If True, the pipeline will first run the HTML copier, then
        Trafilatura, and finally the license annotator. If False, the pipeline will run the license
        annotator first. This was intended for benchmarking. The default (false) is about 10% faster.
        Defaults to False.

    Returns:
        list[PipelineStep]: List of pipeline steps (i.e., the pipeline components)
    """
    if trafiltura_first:
        return [
            WarcReader(
                f"s3://commoncrawl/crawl-data/{dump}/segments/",
                glob_pattern="*/warc/*",
                default_metadata={"dump": dump},
                limit=limit,
            ),
            URLFilter(),
            EmptyTextFilter(),  # filter items with empty HTML (text = read HTML at this point)
            HtmlCopier(),
            Trafilatura(favour_precision=True, timeout=600.0),
            EmptyTextFilter(),  # filter items with empty extracted text
            LanguageFilter(languages=languages, language_threshold=language_threshold),
            LicenseAnnotator(
                html_in_metadata=True,
                remove_html=True,
            ),
            LicenseFilter(),
            JsonlWriter(
                output_folder=f"{output_path}/",
                output_filename="${language}/${rank}.jsonl.gz",
                expand_metadata=True,
            ),
        ]
    else:
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
            LanguageFilter(languages=languages, language_threshold=language_threshold),
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
        ("language_score", pa.float64()),
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
