import json
import re
from collections import Counter, defaultdict
from os import PathLike
from pathlib import Path


def aggregate(pdir: str | PathLike) -> None:
    pdir = Path(pdir)
    dump = pdir.parent.stem
    pfout = pdir.parent / f"{dump}_agg_stats.json"

    stats = {"filter": defaultdict(Counter), "writer": defaultdict(Counter)}
    for pfin in pdir.rglob("*.json"):
        data = json.loads(pfin.read_text(encoding="utf-8"))
        for component in data:
            full_name = re.sub(r"[\ud800-\udfff\ufe0f]", "", component["name"])
            full_name = full_name.split("-", 1)[1]
            comp_type, comp_name = full_name.split(":", 1)
            comp_type = comp_type.strip().lower()
            comp_name = comp_name.strip()

            if comp_type == "filter" and comp_name != "Url-filter":
                stats["filter"][comp_name]["num_input_docs"] += component["stats"]["total"]
                stats["filter"][comp_name]["num_output_docs"] += component["stats"]["forwarded"]
                stats["filter"][comp_name]["num_dropped_docs"] += component["stats"]["dropped"]
            elif comp_type == "writer":
                for item, value in component["stats"].items():
                    if item.endswith(".jsonl.gz"):
                        language = item.split("/")[0]
                        stats["writer"][language] += value

    # Convert defaultdicts and counters to dicts
    for key in stats:
        stats[key] = dict(stats[key])
        for subkey in stats[key]:
            stats[key][subkey] = dict(stats[key][subkey])

    stats = {"dump": dump, **stats}
    with pfout.open("w", encoding="utf-8") as fhout:
        json.dump(stats, fhout, indent=4)


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Aggregate the filter and writer statistics of completed pipeline stats.",
    )
    cparser.add_argument(
        "pdir",
        type=str,
        required=True,
        help="Path to the directory containing the pipeline stats, typically ending in '<dump_name>/stats'."
        " This directory's parent will be used as the dump name. Outputs will be written to this parent"
        " directory as 'agg_stats.json'.",
    )
    cargs = cparser.parse_args()
    aggregate(cargs.pdir)
