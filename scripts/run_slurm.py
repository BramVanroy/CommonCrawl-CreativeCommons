import re
from functools import partial
from pathlib import Path

import yaml
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import (
    LanguageFilter,
    URLFilter,
)
from datatrove.pipeline.readers import JsonlReader, WarcReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from pydantic import BaseModel

from gpt_nl_copyright.components.annotator.html_annotator import HtmlCopier
from gpt_nl_copyright.components.annotator.license_annotator import LicenseAnnotator
from gpt_nl_copyright.components.filters.empty_text_filter import EmptyTextFilter
from gpt_nl_copyright.components.filters.lang_mtd_filter import LanguageMetadataFilter
from gpt_nl_copyright.components.filters.license_filter import LicenseFilter
from gpt_nl_copyright.utils import prepare_for_writing, print_system_stats


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
    writing_time: str
    mem_per_cpu_gb: int = 2
    cpus_per_task: int = 1
    randomize_start_duration: int = 0
    lang_filter_language_threshold: float = 0.65
    languages: list = LANGUAGES
    keep_with_license_only: bool = True


def job_id_retriever(job_id: str) -> str:
    return re.search(r"Submitted batch job (\d+)", job_id).group(1)


def main(
    dump: str,
    output_path: str,
    partition: str,
    pipelines_config: str,
    venv_path: str | None = None,
    account: str | None = None,
):
    print_system_stats()
    config = yaml.safe_load(Path(pipelines_config).read_text(encoding="utf-8"))
    sbatch_args = {"account": account} if account else {}
    cfg = Config(**config)

    all_output_path = f"{output_path}/data/all-unfiltered/"

    main_processing_executor = SlurmPipelineExecutor(
        pipeline=[
            WarcReader(
                f"s3://commoncrawl/crawl-data/{dump}/segments/",
                glob_pattern="*/warc/*",
                default_metadata={"dump": dump},
            ),
            URLFilter(),
            EmptyTextFilter(),  # filter empty HTML
            HtmlCopier(),
            Trafilatura(favour_precision=True, timeout=600.0),
            EmptyTextFilter(),  # filter empty extracted text
            LanguageFilter(languages=cfg.languages, language_threshold=cfg.lang_filter_language_threshold),
            LicenseAnnotator(),
            LicenseFilter(),
            JsonlWriter(output_folder=all_output_path),
        ],
        job_id_retriever=job_id_retriever,
        tasks=cfg.tasks,
        time=cfg.time,
        logging_dir=f"{output_path}/logs/main/",
        slurm_logs_folder=f"{output_path}/slurm-logs/main",
        randomize_start_duration=cfg.randomize_start_duration,  # don't hit the bucket all at once with the list requests
        mem_per_cpu_gb=cfg.mem_per_cpu_gb,
        partition=partition,
        venv_path=venv_path,
        qos="",
        sbatch_args=sbatch_args,
    )
    main_processing_executor.run()

    # Write the results to disk for each language seperately
    reader = JsonlReader(all_output_path)
    for lang in cfg.languages:
        # Write two version to jsonl, one with the HTML removed and one with both text and HTML removed
        no_text_no_html = partial(prepare_for_writing, output_text=False, output_html=False)
        no_html = partial(prepare_for_writing, output_text=True, output_html=False)

        SlurmPipelineExecutor(
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
            job_id_retriever=job_id_retriever,
            logging_dir=f"{output_path}/logs/lang-writer-{lang}/",
            slurm_logs_folder=f"{output_path}/slurm-logs/lang-writer-{lang}/",
            depends=main_processing_executor,
            tasks=cfg.tasks,
            time=cfg.writing_time,
            randomize_start_duration=cfg.randomize_start_duration,
            mem_per_cpu_gb=cfg.mem_per_cpu_gb,
            partition=partition,
            venv_path=venv_path,
            qos="",
            sbatch_args=sbatch_args,
        ).run()


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
