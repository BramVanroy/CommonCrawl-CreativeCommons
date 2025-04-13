import pytest
from commoncrawl_cc_annotation.script_utils import get_dumps_with_duckdb

@pytest.mark.parametrize(
    "dump_name, ignore_duckdb_for, languages, expected_ignore_duckdb_for, expected_ignore_all_duckdb",
    [
        # Test case 1: Dump is older than 2024-18, no languages ignored initially
        (
            "CC-MAIN-2023-10", 
            [], 
            ["eng_Latn", "fra_Latn"], 
            [], 
            False
        ),
        # Test case 2: Dump is newer than 2024-18, non-English languages should be ignored
        (
            "CC-MAIN-2024-19", 
            [], 
            ["eng_Latn", "fra_Latn"], 
            ["fra_Latn"], 
            False
        ),
        # Test case 3: Dump is newer than 2024-51, English should also be ignored
        (
            "CC-MAIN-2024-52", 
            [], 
            ["eng_Latn", "fra_Latn"], 
            ["eng_Latn", "fra_Latn"], 
            True
        ),
        # Test case 4: Some languages already ignored, additional ignoring based on dump
        (
            "CC-MAIN-2024-19", 
            ["spa_Latn"], 
            ["eng_Latn", "fra_Latn", "spa_Latn"], 
            ["spa_Latn", "fra_Latn"], 
            False
        ),
        # Test case 5: All languages ignored initially by the user
        (
            "CC-MAIN-2024-10", 
            ["eng_Latn", "fra_Latn"], 
            ["eng_Latn", "fra_Latn"], 
            ["eng_Latn", "fra_Latn"], 
            True
        ),
    ],
)
def test_get_dumps_with_duckdb(
    dump_name, ignore_duckdb_for, languages, expected_ignore_duckdb_for, expected_ignore_all_duckdb
):
    result_ignore_duckdb_for, result_ignore_all_duckdb = get_dumps_with_duckdb(
        dump_name, languages, ignore_duckdb_for
    )
    assert set(result_ignore_duckdb_for) == set(expected_ignore_duckdb_for)
    assert result_ignore_all_duckdb == expected_ignore_all_duckdb