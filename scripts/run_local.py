from pathlib import Path

import yaml
from datatrove.executor.local import LocalPipelineExecutor

from c5.script_utils import (
    BaseConfig,
    build_containment_pipeline,
    build_main_pipeline,
    download_duckdbs,
    get_fw_c_and_d_domains,
)
from c5.utils import PROJECT_ROOT


def main(
    dump: str,
    output_path: str,
    pipelines_config: str,
):
    if pipelines_config and Path(pipelines_config).is_file():
        config = yaml.safe_load(Path(pipelines_config).read_text(encoding="utf-8"))
    else:
        raise FileNotFoundError("pipelines_config file not found")

    cfg = BaseConfig(**config)

    fw_duckdb_path = cfg.fw_duckdb_templ_path.format(dump=dump)
    ignore_duckdb_for, _ = download_duckdbs(dump, fw_duckdb_path, cfg)

    main_output_path = output_path.rstrip("/") + "-main/"
    main_dump_output_path = main_output_path + dump + "/"
    main_pipeline = build_main_pipeline(
        dump=dump,
        output_folder=main_dump_output_path,
        languages=cfg.languages,
        ignore_undetermined=cfg.ignore_undetermined,
        limit=cfg.limit,
        extra_domains=get_fw_c_and_d_domains(),
    )
    main_executor = LocalPipelineExecutor(
        pipeline=main_pipeline,
        tasks=cfg.main_tasks,
        workers=cfg.main_workers,
        logging_dir=str(PROJECT_ROOT / "logs" / "main-logs" / dump),
        randomize_start_duration=cfg.randomize_start_duration,
    )

    # Do containment checking (separately because it's intensive on storage)
    containment_dump_output_path = output_path.rstrip("/") + "/" + dump + "/"
    for language in cfg.languages:
        is_fw2 = language != "eng_Latn"
        ignore_duckdb = language in ignore_duckdb_for

        if is_fw2:
            duckdb_path = cfg.fw2_duckdb_templ_path.format(dump=dump, language=language)
        else:
            duckdb_path = cfg.fw_duckdb_templ_path.format(dump=dump)

        containment_pipeline = build_containment_pipeline(
            input_path=main_dump_output_path + language + "/",
            duckdb_path=duckdb_path,
            is_fw2=is_fw2,
            overwrite_with_none=ignore_duckdb,
            output_folder=containment_dump_output_path,
        )
        containment_executor = LocalPipelineExecutor(
            pipeline=containment_pipeline,
            tasks=cfg.containment_tasks,
            workers=cfg.containment_workers,
            logging_dir=str(PROJECT_ROOT / "logs" / "containment-logs" / dump / language),
            depends=main_executor,
        )

        containment_executor.run()


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
    cparser.add_argument(
        "-c",
        "--pipelines_config",
        type=str,
        help="Path to the pipelines YAML config file.",
    )
    cparser.add_argument(
        "-o",
        "--output_path",
        default="output/",
        type=str,
        help="Output path for the pipeline. Output will be saved in a subdirectory named after the dump.",
    )

    cli_kwargs = vars(cparser.parse_args())
    main(**cli_kwargs)
