from pathlib import Path

import yaml

from c5.script_utils import SlurmConfig, SlurmUploadConfig


def all_files_accounted_for(pdir: Path, num_files: int) -> list[int]:
    # Generate zero filled file names based on the expected number of files, e.g. 03784
    expected_fnames = {f"{i:05d}" for i in range(num_files)}
    actual_fnames = {p.name for p in pdir.iterdir() if p.is_file()}

    # Return the set of missing file names
    return list(sorted(expected_fnames - actual_fnames))


def main(config_file: str, upload_config_file: str, log_dir: str, crawl_name: str):
    """
    Check which parts of the processing pipeline have fully completed. Checks
    the main logs, containment logs and upload logs.

    :param config_file: Path to the configuration file.
    :param log_dir: Directory where logs are stored.
    :param crawl_name: Name of the crawl to check.
    """
    pfconfig = Path(config_file)
    cfg = yaml.safe_load(pfconfig.read_text(encoding="utf-8"))
    cfg = SlurmConfig(**cfg)

    plog_dir = Path(log_dir)

    # Main tasks
    num_main_tasks = cfg.main_tasks
    main_completions = plog_dir / "main-logs" / crawl_name / "completions"
    if not main_completions.exists():
        raise FileNotFoundError(f"[ERROR] Main completions directory {main_completions} does not exist.")

    if missing_files := all_files_accounted_for(main_completions, num_main_tasks):
        print(f"[ERROR] Main tasks missing these files: {missing_files}")
    else:
        print("[INFO] All main tasks completed successfully.")

    # Containment tasks
    num_containment_tasks = cfg.containment_tasks
    languages = cfg.languages
    containment_tasks = plog_dir / "containment-logs" / crawl_name
    lang_dirs = [containment_tasks / lang for lang in languages]
    had_error = False
    for lang_dir in lang_dirs:
        if not lang_dir.exists():
            had_error = True
            print(f"[WARNING] Language directory {lang_dir} does not exist. Likely to be problematic (though for very tiny minority languages it is technically possible to not have any results).")
            continue
        completions_dir = lang_dir / "completions"
        if not completions_dir.exists():
            had_error = True
            print(f"[ERROR] Completions directory {completions_dir} does not exist for language {lang_dir.name}.")
        elif missing_files := all_files_accounted_for(completions_dir, num_containment_tasks):
            print(f"[WARNING] Containment tasks for {lang_dir.name} missing these files: {missing_files}")
            had_error = True
    
    if had_error:
        print("[ERROR] Some containment tasks had issues. Please check the warnings above.")
    else:
        print("[INFO] All containment tasks completed successfully for all languages.")
        
    # Upload tasks
    pf_upl_config = Path(upload_config_file)
    upl_cfg = yaml.safe_load(pf_upl_config.read_text(encoding="utf-8"))
    upl_cfg = SlurmUploadConfig(**upl_cfg)
    num_upload_tasks = upl_cfg.tasks
    upload_tasks = plog_dir / "upload-logs" / crawl_name / "completions"
    if not upload_tasks.exists():
        raise FileNotFoundError(f"[ERROR] Upload completions directory {upload_tasks} does not exist. Uploading not started yet?")
    if missing_files := all_files_accounted_for(upload_tasks, num_upload_tasks):
        print(f"[ERROR] Upload tasks missing these files: {missing_files}")
    else:
        print("[INFO] All upload tasks completed successfully.")

if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(description="Check completed crawls.")
    cparser.add_argument("-c", "--config_file", type=str, required=True, help="Path to the configuration file.")
    cparser.add_argument(
        "-u", "--upload_config_file", type=str, required=True, help="Path to the configuration file for uploading"
    )
    cparser.add_argument("-d", "--log_dir", type=str, required=True, help="Directory where logs are stored.")
    cparser.add_argument(
        "-n", "--crawl_name", type=str, required=True, help="Name of the crawl to check, e.g. CC-MAIN-2020-05"
    )
    cargs = cparser.parse_args()
    main(**vars(cargs))
