from pathlib import Path

import yaml
from datatrove.executor.local import LocalPipelineExecutor

from c5.script_utils import BaseUploadConfig, build_upload_pipeline
from c5.utils import PROJECT_ROOT


def main(
    jsonl_path: str,
    output_path: str,
    hf_repo: str,
    pipelines_config: str | None = None,
):
    if Path(jsonl_path).stem != Path(output_path).stem:
        raise ValueError("JSONL path and output path must both end in the same dump name.")

    crawl_name = Path(output_path).stem
    if pipelines_config and Path(pipelines_config).is_file():
        config = yaml.safe_load(Path(pipelines_config).read_text(encoding="utf-8"))
    else:
        config = {}
    cfg = BaseUploadConfig(**config)

    pipeline = build_upload_pipeline(
        jsonl_path=jsonl_path,
        output_path=output_path,
        hf_repo=hf_repo,
        limit=cfg.limit,
    )
    log_dir = str(PROJECT_ROOT / "logs" / "upload-logs" / crawl_name)
    LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=cfg.tasks,
        workers=cfg.workers,
        logging_dir=log_dir,
        randomize_start_duration=cfg.randomize_start_duration,
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
    cparser.add_argument(
        "-c",
        "--pipelines_config",
        default=None,
        type=str,
        help="Path to the pipelines YAML config file. If not given will use default values.",
    )

    cli_kwargs = vars(cparser.parse_args())
    main(**cli_kwargs)
