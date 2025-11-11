from pathlib import Path

import yaml

from c5.components.slurm_executor import C5SlurmExecutor
from c5.script_utils import SlurmUploadConfig, build_upload_pipeline, job_id_retriever
from c5.utils import PROJECT_ROOT, print_system_stats


def main(
    jsonl_path: str,
    output_path: str,
    hf_repo: str,
    partition: str,
    pipelines_config: str,
    venv_path: str | None = None,
    account: str | None = None,
):
    if Path(jsonl_path).stem != Path(output_path).stem:
        raise ValueError("JSONL path and output path must both end in the same dump name.")

    crawl_name = Path(output_path).stem
    print_system_stats()
    config = yaml.safe_load(Path(pipelines_config).read_text(encoding="utf-8"))
    sbatch_args = {"account": account} if account else {}
    cfg = SlurmUploadConfig(**config)

    pipeline = build_upload_pipeline(
        jsonl_path=jsonl_path,
        output_path=output_path,
        hf_repo=hf_repo,
        limit=cfg.limit,
    )
    log_dir = str(PROJECT_ROOT / "logs" / "upload-logs" / crawl_name)
    slurm_log_dir = str(PROJECT_ROOT / "slurm-logs" / "upload-logs" / crawl_name)
    C5SlurmExecutor(
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
        job_name="upload",
    ).run()


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cparser.add_argument(
        "-j",
        "--jsonl_path",
        type=str,
        required=True,
        help="Path to the directory containing the JSONL files from processing a single dump, most top-level, e.g. `output/CC-MAIN-2019-30`. Must end in dump name.",
    )
    cparser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=True,
        help="Output path to save the parquet files before uploading, e.g. `parquet_output/CC-MAIN-2019-30`. Must end in dump name.",
    )
    cparser.add_argument("-r", "--hf_repo", type=str, required=True, help="Hugging Face repository name")
    cparser.add_argument("--partition", type=str, required=True, help="Slurm partition")
    cparser.add_argument("--pipelines_config", type=str, required=True, help="Path to the pipelines YAML config file")
    cparser.add_argument("--venv_path", type=str, help="Path to the virtual environment")
    cparser.add_argument("--account", type=str, help="Slurm account")

    cli_kwargs = vars(cparser.parse_args())
    main(**cli_kwargs)
