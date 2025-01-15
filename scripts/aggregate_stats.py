import json
import re
from collections import Counter, defaultdict
from os import PathLike
from pathlib import Path


def aggregate(pdir: str | PathLike, verbose: bool = False) -> None:
    pdir = Path(pdir)
    dump = pdir.parent.stem
    pfout = pdir.parent / f"{dump}_agg_stats.json"

    stats = {"filter": defaultdict(Counter), "writer": Counter}
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
                if "dropped" in component["stats"]:
                    stats["filter"][comp_name]["num_dropped_docs"] += component["stats"]["dropped"]
            elif comp_type == "writer":
                for item, value in component["stats"].items():
                    if item.endswith(".jsonl.gz"):
                        language = item.split("/")[0]
                        stats["writer"][language] += value

    # Convert defaultdicts and counters to dicts
    stats["filter"] = dict(stats["filter"])
    stats["filter"] = {comp_name: dict(counter) for comp_name, counter in stats["filter"].items()}
    stats["writer"] = dict(stats["writer"])

    stats = {"dump": dump, **stats}
    with pfout.open("w", encoding="utf-8") as fhout:
        json.dump(stats, fhout, indent=4)

    if verbose:
        print(json.dumps(stats, indent=4))


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Aggregate the filter and writer statistics of completed pipeline stats.",
    )
    cparser.add_argument(
        "pdir",
        type=str,
        help="Path to the directory containing the pipeline stats, typically ending in '<dump_name>/stats'."
        " This directory's parent will be used as the dump name. Outputs will be written to this parent"
        " directory as 'agg_stats.json'.",
    )
    cparser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print the aggregated statistics to stdout.",
    )
    cargs = cparser.parse_args()
    aggregate(cargs.pdir, cargs.verbose)
