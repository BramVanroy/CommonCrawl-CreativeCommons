"""
Heavily inspired by https://raw.githubusercontent.com/huggingface/datatrove/main/examples/fineweb.py
"""

from pathlib import Path

import yaml
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import (
    LanguageFilter,
    URLFilter,
)
from datatrove.pipeline.readers import WarcReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from pydantic import BaseModel

from gpt_nl_copyright.components.copyright import CopyrightAnnotator
from gpt_nl_copyright.components.html_copier import HtmlCopier


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


def main(
    dump: str,
    output_path: str,
    pipelines_config: str | None = None,
):
    if pipelines_config and Path(pipelines_config).is_file():
        config = yaml.safe_load(Path(pipelines_config).read_text(encoding="utf-8"))
    else:
        config = {}
    extract_cfg = Config(**config)

    d_base_filter = f"{output_path}/base_processing"

    main_processing_executor = LocalPipelineExecutor(
        pipeline=[
            WarcReader(
                f"s3://commoncrawl/crawl-data/{dump}/segments/",
                glob_pattern="*/warc/*",
                default_metadata={"dump": dump},
            ),
            URLFilter(),
            HtmlCopier(),
            Trafilatura(favour_precision=True, timeout=600.0),
            LanguageFilter(
                languages=extract_cfg.languages, language_threshold=extract_cfg.lang_filter_language_threshold
            ),
            CopyrightAnnotator(),
            JsonlWriter(f"{d_base_filter}/output/{dump}", expand_metadata=True),
        ],
        tasks=extract_cfg.tasks,
        workers=extract_cfg.workers,
        logging_dir=f"{output_path}/logs/base_processing/{dump}",
        randomize_start_duration=extract_cfg.randomize_start_duration,
    )
    main_processing_executor.run()


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
