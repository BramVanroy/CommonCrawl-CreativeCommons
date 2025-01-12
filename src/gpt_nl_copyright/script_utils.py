# Dutch, Frisian, English, Spanish, French, Italian, German, Afrikaans
import dataclasses
from functools import partial

from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import (
    LanguageFilter,
    URLFilter,
)
from datatrove.pipeline.readers import JsonlReader, WarcReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from pydantic import BaseModel

from gpt_nl_copyright.components.annotator.license_annotator import LicenseAnnotator
from gpt_nl_copyright.components.filters.empty_text_filter import EmptyTextFilter
from gpt_nl_copyright.components.filters.lang_mtd_filter import LanguageMetadataFilter
from gpt_nl_copyright.components.filters.license_filter import LicenseFilter


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
    lang_filter_language_threshold: float = 0.65
    languages: list = LANGUAGES
    keep_with_license_only: bool = True


class LocalConfig(_BaseConfig):
    workers: int = -1


class SlurmConfig(_BaseConfig):
    time: str
    writing_time: str
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


def build_main_pipeline(
    dump: str, all_output_path: str, languages: list[str], language_threshold: float = 0.65
) -> list[PipelineStep]:
    return [
        WarcReader(
            f"s3://commoncrawl/crawl-data/{dump}/segments/",
            glob_pattern="*/warc/*",
            default_metadata={"dump": dump},
            limit=6,
        ),
        URLFilter(),
        EmptyTextFilter(),  # filter empty HTML
        LicenseAnnotator(),
        LicenseFilter(),
        Trafilatura(favour_precision=True, timeout=600.0),
        EmptyTextFilter(),  # filter empty extracted text
        LanguageFilter(languages=languages, language_threshold=language_threshold),
        JsonlWriter(output_folder=all_output_path),
    ]


def build_lang_writer_pipeline(reader: JsonlReader, lang: str, output_path: str) -> list[PipelineStep]:
    # Write two version to jsonl, one with the HTML removed and one with both text and HTML removed
    no_text_no_html = partial(prepare_for_writing, output_text=False, output_html=False)
    no_html = partial(prepare_for_writing, output_text=True, output_html=False)

    return (
        [
            reader,
            LanguageMetadataFilter(lang),
            JsonlWriter(
                output_folder=f"{output_path}/data/{lang}/no_text_no_html/",
                adapter=no_text_no_html,
            ),
            JsonlWriter(
                output_folder=f"{output_path}/data/{lang}/no_html/",
                adapter=no_html,
            ),
        ],
    )
