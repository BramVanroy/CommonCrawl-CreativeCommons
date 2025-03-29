from pathlib import Path

import yaml
from datatrove.executor.slurm import SlurmPipelineExecutor

from commoncrawl_cc_annotation.script_utils import (
    SlurmConfig,
    auto_download_duckdbs,
    build_containment_pipeline,
    build_main_pipeline,
    get_fw_c_and_d_domains,
    job_id_retriever,
)
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

    # If dump is more recent than 2024-18, we auto-set the found_in_fw
    # column to False because FW2 does not have that data. That avoids the expensive
    # containment checking.
    dump_year = int(dump.split("-")[2])
    dump_issue = int(dump.split("-")[3])

    # Not in FineWeb-2
    ignore_duckdb_for = cfg.ignore_duckdb_for or []
    if dump_year > 2024 or (dump_year == 2024 and dump_issue > 18):
        ignore_duckdb_for += cfg.languages

    # FW1 v1.3 contains data up to 2024-51
    if dump_year > 2024 or (dump_year == 2024 and dump_issue > 51):
        ignore_duckdb_for += "eng_Latn"

    if "{language}" not in cfg.fw2_duckdb_templ_path:
        raise ValueError("The fw2_duckdb_templ_path must contain the string '{language}'")

    if "{dump}" not in cfg.fw_duckdb_templ_path:
        raise ValueError("The fw_duckdb_templ_path must contain the string '{dump}'")

    fw_duckdb_path = cfg.fw_duckdb_templ_path.format(dump=dump)

    # Only download languages that are not in ignore_duckdb_for (occurs when crawl is too recent)
    duckdb_languages = []
    for lang in cfg.languages:
        if lang not in ignore_duckdb_for:
            duckdb_languages.append(lang)

    if duckdb_languages:
        auto_download_duckdbs(
            languages=duckdb_languages,
            fw2_duckdb_templ_path=cfg.fw2_duckdb_templ_path,
            fw_duckdb_path=fw_duckdb_path,
        )

    main_output_path = output_path.rstrip("/") + "-main/"
    main_dump_output_path = main_output_path + dump + "/"
    main_pipeline = build_main_pipeline(
        dump=dump,
        output_path=main_dump_output_path,
        languages=cfg.languages,
        language_threshold=cfg.language_threshold,
        limit=cfg.limit,
        extra_domains=get_fw_c_and_d_domains(),
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
    dump_output_path = output_path.rstrip("/") + "/" + dump + "/"
    containment_pipeline = build_containment_pipeline(
        fw_duckdb_path=fw_duckdb_path,
        fw2_duckdb_templ_path=cfg.fw2_duckdb_templ_path,
        ignore_duckdb_for=cfg.ignore_duckdb_for,
        input_path=main_dump_output_path,
        output_path=dump_output_path,
        overwrite_with_none=cfg.overwrite_with_none,
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
