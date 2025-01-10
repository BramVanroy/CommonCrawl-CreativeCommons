"""
Heavily inspired by https://raw.githubusercontent.com/huggingface/datatrove/main/examples/fineweb.py
"""

from functools import partial
from pathlib import Path

import yaml
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import (
    LanguageFilter,
    URLFilter,
)
from datatrove.pipeline.readers import JsonlReader, WarcReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from pydantic import BaseModel

from gpt_nl_copyright.components.annotator.copyright_annotator import CopyrightAnnotator
from gpt_nl_copyright.components.annotator.html_annotator import HtmlCopier
from gpt_nl_copyright.components.filters.empty_text_filter import EmptyTextFilter
from gpt_nl_copyright.components.filters.lang_mtd_filter import LanguageMetadataFilter
from gpt_nl_copyright.components.filters.license_filter import CopyrightFilter
from gpt_nl_copyright.utils import prepare_for_writing


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


class Config(BaseModel):
    tasks: int = 1
    workers: int = -1
    randomize_start_duration: int = 0
    lang_filter_language_threshold: float = 0.65
    languages: list = LANGUAGES
    keep_with_license_only: bool = True


def main(
    dump: str,
    output_path: str,
    pipelines_config: str | None = None,
):
    if pipelines_config and Path(pipelines_config).is_file():
        config = yaml.safe_load(Path(pipelines_config).read_text(encoding="utf-8"))
    else:
        config = {}
    cfg = Config(**config)

    all_output_path = f"{output_path}/data/all-unfiltered/"

    main_processing_executor = LocalPipelineExecutor(
        pipeline=[
            WarcReader(
                f"s3://commoncrawl/crawl-data/{dump}/segments/",
                glob_pattern="*/warc/*",
                default_metadata={"dump": dump},
                # limit=100,
            ),
            URLFilter(),
            EmptyTextFilter(),  # filter empty HTML
            HtmlCopier(),
            Trafilatura(favour_precision=True, timeout=600.0),
            EmptyTextFilter(),  # filter empty extracted text
            LanguageFilter(languages=cfg.languages, language_threshold=cfg.lang_filter_language_threshold),
            CopyrightAnnotator(),
            CopyrightFilter(),
            JsonlWriter(output_folder=all_output_path),
        ],
        tasks=cfg.tasks,
        workers=cfg.workers,
        logging_dir=f"{output_path}/logs/main/",
        randomize_start_duration=cfg.randomize_start_duration,
    )
    main_processing_executor.run()

    # Write the results to disk for each language seperately
    reader = JsonlReader(all_output_path)
    for lang in cfg.languages:
        # Write two version to jsonl, one with the HTML removed and one with both text and HTML removed
        no_text_no_html = partial(prepare_for_writing, output_text=False, output_html=False)
        no_html = partial(prepare_for_writing, output_text=True, output_html=False)

        LocalPipelineExecutor(
            pipeline=[
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
            tasks=cfg.tasks,
            workers=cfg.workers,
            logging_dir=f"{output_path}/logs/lang-writer-{lang}/",
            depends=main_processing_executor,
        ).run()


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cparser.add_argument(
        "-d",
        "--dump",
        type=str,
        required=True,
        help="CommonCrawl dump, e.g. 'CC-MAIN-2024-51' (see https://commoncrawl.org/overview)",
    )
    cparser.add_argument("-o", "--output_path", type=str, required=True, help="Output path")
    cparser.add_argument(
        "-c",
        "--pipelines_config",
        default=None,
        type=str,
        help="Path to the pipelines YAML config file. If not given will use default values.",
    )

    cli_kwargs = vars(cparser.parse_args())
    main(**cli_kwargs)
