import re
from pathlib import Path

import yaml
from datatrove.executor.slurm import SlurmPipelineExecutor

from gpt_nl_copyright.script_utils import SlurmConfig, build_pipeline
from gpt_nl_copyright.utils import print_system_stats


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
    cfg = SlurmConfig(**config)

    pipeline = build_pipeline(
        dump=dump,
        output_path=output_path,
        languages=cfg.languages,
        language_threshold=cfg.language_threshold,
        limit=cfg.limit,
    )
    main_processing_executor = SlurmPipelineExecutor(
        pipeline=pipeline,
        job_id_retriever=job_id_retriever,
        tasks=cfg.tasks,
        time=cfg.time,
        logging_dir=f"{output_path}/logs/",
        slurm_logs_folder=f"{output_path}/slurm-logs/",
        randomize_start_duration=cfg.randomize_start_duration,
        mem_per_cpu_gb=cfg.mem_per_cpu_gb,
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
