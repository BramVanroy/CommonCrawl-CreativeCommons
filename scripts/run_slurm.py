"""
This file contains the code used to process and create the
ReinWeb dataset. Heavily inspired by https://raw.githubusercontent.com/huggingface/datatrove/main/examples/fineweb.py
"""

"""
This file contains the code used to process and create the
ReinWeb dataset. Heavily inspired by https://raw.githubusercontent.com/huggingface/datatrove/main/examples/fineweb.py
"""

from pathlib import Path

import yaml
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import (
    LanguageFilter,
    URLFilter,
)
from datatrove.pipeline.readers import WarcReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from pydantic import BaseModel

from gpt_nl_copyright.components.copyright import CopyrightAnnotator


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
    tasks: int
    time: str
    mem_per_cpu_gb: int = 2
    cpus_per_task: int = 1
    randomize_start_duration: int = 0
    do_pipeline: bool = True
    lang_filter_language_threshold: float = 0.65
    languages: list = LANGUAGES


def main(
    dump: str,
    output_path: str,
    partition: str,
    pipelines_config: str,
    venv_path: str | None = None,
    account: str | None = None,
):
    config = yaml.safe_load(Path(pipelines_config).read_text(encoding="utf-8"))
    sbatch_args = {"account": account} if account else {}

    extract_cfg = Config(**config)

    d_base_filter = f"{output_path}/base_processing"

    main_processing_executor = SlurmPipelineExecutor(
        job_name=f"cc_{dump}",
        pipeline=[
            WarcReader(
                f"s3://commoncrawl/crawl-data/{dump}/segments/",
                glob_pattern="*/warc/*",
                default_metadata={"dump": dump},
            ),
            URLFilter(),
            CopyrightAnnotator(),
            Trafilatura(favour_precision=True, timeout=600.0),
            LanguageFilter(
                languages=extract_cfg.languages, language_threshold=extract_cfg.lang_filter_language_threshold
            ),
            JsonlWriter(f"{d_base_filter}/output/{dump}"),
        ],
        tasks=extract_cfg.tasks,
        time=extract_cfg.time,
        logging_dir=f"{output_path}/logs/base_processing/{dump}",
        slurm_logs_folder=f"slurm-logs/base_processing/{dump}/slurm_logs",  # must be local
        randomize_start_duration=extract_cfg.randomize_start_duration,  # don't hit the bucket all at once with the list requests
        mem_per_cpu_gb=extract_cfg.mem_per_cpu_gb,
        partition=partition,
        venv_path=venv_path,
        qos="",
        sbatch_args=sbatch_args,
    )
    main_processing_executor.run()


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cparser.add_argument(
        "--dump", type=str, required=True, help="CommonCrawl dump (see https://commoncrawl.org/overview)"
    )
    cparser.add_argument("--output_path", type=str, required=True, help="Output path")
    cparser.add_argument("--partition", type=str, required=True, help="Slurm partition")
    cparser.add_argument("--pipelines_config", type=str, required=True, help="Path to the pipelines YAML config file")
    cparser.add_argument("--venv_path", type=str, help="Path to the virtual environment")
    cparser.add_argument("--account", type=str, help="Slurm account")

    cli_kwargs = vars(cparser.parse_args())
    main(**cli_kwargs)
