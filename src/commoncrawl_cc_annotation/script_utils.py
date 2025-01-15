from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import (
    LanguageFilter,
    URLFilter,
)
from datatrove.pipeline.readers import WarcReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from pydantic import BaseModel

from commoncrawl_cc_annotation.components.annotators.license_annotator import LicenseAnnotator
from commoncrawl_cc_annotation.components.filters.empty_text_filter import EmptyTextFilter
from commoncrawl_cc_annotation.components.filters.license_filter import LicenseFilter


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
    dump: str, output_path: str, languages: list[str], language_threshold: float = 0.65, limit: int = -1
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

    Returns:
        list[PipelineStep]: List of pipeline steps (i.e., the pipeline components)
    """
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
        LanguageFilter(languages=languages, language_threshold=language_threshold),
        EmptyTextFilter(),  # filter items with empty extracted text
        JsonlWriter(
            output_folder=f"{output_path}/",
            output_filename="${language}/${rank}.jsonl.gz",
            expand_metadata=True,
        ),
    ]


if __name__ == "__main__":
    _BaseConfig()
