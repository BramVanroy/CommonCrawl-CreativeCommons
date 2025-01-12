import dataclasses
from functools import partial

from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import (
    LanguageFilter,
    URLFilter,
)
from datatrove.pipeline.readers import WarcReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from pydantic import BaseModel

from gpt_nl_copyright.components.annotator.license_annotator import LicenseAnnotator
from gpt_nl_copyright.components.filters.empty_text_filter import EmptyTextFilter
from gpt_nl_copyright.components.filters.license_filter import LicenseFilter


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
    tasks: int = 1
    randomize_start_duration: int = 0
    language_threshold: float = 0.65
    languages: list = LANGUAGES
    keep_with_license_only: bool = True
    limit: int = -1


class LocalConfig(_BaseConfig):
    workers: int = -1


class SlurmConfig(_BaseConfig):
    time: str
    mem_per_cpu_gb: int = 2
    cpus_per_task: int = 1


def prepare_for_writing(self, document: Document, output_text: bool = True, output_html: bool = False) -> dict:
    """
    Potentially remove text and/or html from the document before writing to disk. The 'self' is needed
    because this function is passed to the JsonlWriter as an adapter where it will be turned into a method.
    Args:
        document: document to format
        output_text: whether to include the text in the output
        output_html: whether to include the html in the output

    Returns: a dictionary to write to disk

    """
    data = {key: val for key, val in dataclasses.asdict(document).items() if val}
    # Expand metadata into its own columns
    data |= data.pop("metadata")

    if not output_text:
        data.pop("text", None)
    if not output_html:
        data.pop("html", None)

    return data


def build_pipeline(
    dump: str, output_path: str, languages: list[str], language_threshold: float = 0.65, limit: int = -1
) -> list[PipelineStep]:
    # Write two version to jsonl, one with the HTML removed and one with both text and HTML removed
    no_text_no_html = partial(prepare_for_writing, output_text=False, output_html=False)
    no_html = partial(prepare_for_writing, output_text=True, output_html=False)
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
            output_folder=f"{output_path}/data/without_text/",
            adapter=no_text_no_html,
            output_filename="${language}/${rank}.jsonl.gz",
        ),
        JsonlWriter(
            output_folder=f"{output_path}/data/with_text/",
            adapter=no_html,
            output_filename="${language}/${rank}.jsonl.gz",
        ),
    ]
