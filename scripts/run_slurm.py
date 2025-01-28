from pathlib import Path

import yaml
from datatrove.executor.slurm import SlurmPipelineExecutor

from commoncrawl_cc_annotation.script_utils import SlurmConfig, build_pipeline, job_id_retriever
from commoncrawl_cc_annotation.utils import PROJECT_ROOT, print_system_stats


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
        duckdb_templ_path=cfg.duckdb_templ_path,
        ignore_duckdb_for=cfg.ignore_duckdb_for,
        languages=cfg.languages,
        language_threshold=cfg.language_threshold,
        limit=cfg.limit,
    )
    log_dir = str(PROJECT_ROOT / "logs" / dump)
    slurm_log_dir = str(PROJECT_ROOT / "slurm-logs" / dump)
    SlurmPipelineExecutor(
        pipeline=pipeline,
        job_id_retriever=job_id_retriever,
        tasks=cfg.tasks,
        time=cfg.time,
        logging_dir=log_dir,
        slurm_logs_folder=slurm_log_dir,
        randomize_start_duration=cfg.randomize_start_duration,
        mem_per_cpu_gb=cfg.mem_per_cpu_gb,
        partition=partition,
        venv_path=venv_path,
        qos="",
        sbatch_args=sbatch_args,
        job_name="process",
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
