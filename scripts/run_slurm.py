from pathlib import Path

import yaml
from datatrove.executor.slurm import SlurmPipelineExecutor
from commoncrawl_cc_annotation.script_utils import SlurmConfig, build_main_pipeline, build_containment_pipeline, job_id_retriever
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

    main_output_path = output_path.rstrip("/") + "-main/"
    main_pipeline = build_main_pipeline(
        dump=dump,
        output_path=main_output_path,
        languages=cfg.languages,
        language_threshold=cfg.language_threshold,
        limit=cfg.limit,
    )
    main_executor = SlurmPipelineExecutor(
        pipeline=main_pipeline,
        job_id_retriever=job_id_retriever,
        tasks=cfg.main_tasks,
        workers=cfg.main_workers,
        time=cfg.main_time,
        logging_dir=str(PROJECT_ROOT / "logs" / "main-logs" / dump),
        slurm_logs_folder=str(PROJECT_ROOT / "slurm-logs" / "main-logs" / dump),
        randomize_start_duration=cfg.randomize_start_duration,
        mem_per_cpu_gb=cfg.main_mem_per_cpu_gb,
        cpus_per_task=cfg.main_cpus_per_task,
        partition=partition,
        venv_path=venv_path,
        qos="",
        sbatch_args=sbatch_args,
        job_name="process-main",
    )

    # Do containment checking (separately because it's intensive on storage)
    containment_pipeline = build_containment_pipeline(
        duckdb_templ_path=cfg.duckdb_templ_path,
        ignore_duckdb_for=cfg.ignore_duckdb_for,
        input_path=main_output_path,
        output_path=output_path,
    )
    containment_executor = SlurmPipelineExecutor(
        pipeline=containment_pipeline,
        job_id_retriever=job_id_retriever,
        tasks=cfg.containment_tasks,
        workers=cfg.containment_workers,
        time=cfg.containment_time,
        logging_dir=str(PROJECT_ROOT / "logs" / "containment-logs" / dump),
        slurm_logs_folder=str(PROJECT_ROOT / "slurm-logs" / "containment-logs" / dump),
        mem_per_cpu_gb=cfg.containment_mem_per_cpu_gb,
        cpus_per_task=cfg.containment_cpus_per_task,
        partition=partition,
        venv_path=venv_path,
        qos="",
        sbatch_args=sbatch_args,
        job_name="process-containment",
        depends=main_executor,
    )

    containment_executor.run()



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
