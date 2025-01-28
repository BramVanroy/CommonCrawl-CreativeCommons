from pathlib import Path

import yaml
from datatrove.executor.local import LocalPipelineExecutor

from commoncrawl_cc_annotation.script_utils import LocalConfig, build_pipeline
from commoncrawl_cc_annotation.utils import PROJECT_ROOT


def main(
    dump: str,
    output_path: str,
    pipelines_config: str | None = None,
):
    if pipelines_config and Path(pipelines_config).is_file():
        config = yaml.safe_load(Path(pipelines_config).read_text(encoding="utf-8"))
    else:
        config = {}
    cfg = LocalConfig(**config)

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
