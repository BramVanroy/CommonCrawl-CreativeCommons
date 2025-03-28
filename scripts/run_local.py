from pathlib import Path

import yaml
from datatrove.executor.local import LocalPipelineExecutor

from commoncrawl_cc_annotation.script_utils import (
    BaseConfig,
    auto_download_duckdbs,
    build_containment_pipeline,
    build_main_pipeline,
    get_fw_c_and_d_domains,
)
from commoncrawl_cc_annotation.utils import PROJECT_ROOT


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

    auto_download_duckdbs(
        languages=cfg.languages,
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
    main_executor = LocalPipelineExecutor(
        pipeline=main_pipeline,
        tasks=cfg.main_tasks,
        workers=cfg.main_workers,
        logging_dir=str(PROJECT_ROOT / "logs" / "main-logs" / dump),
        randomize_start_duration=cfg.randomize_start_duration,
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
    containment_executor = LocalPipelineExecutor(
        pipeline=containment_pipeline,
        tasks=cfg.containment_tasks,
        workers=cfg.containment_workers,
        logging_dir=str(PROJECT_ROOT / "logs" / "containment-logs" / dump),
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
    cparser.add_argument("-o", "--output_path", type=str, required=True, help="Output path")
    cparser.add_argument(
        "-c",
        "--pipelines_config",
        type=str,
        help="Path to the pipelines YAML config file.",
    )

    cli_kwargs = vars(cparser.parse_args())
    main(**cli_kwargs)
