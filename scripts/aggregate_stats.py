import json
import re
from collections import Counter, defaultdict
from os import PathLike
from pathlib import Path
from tqdm import tqdm

def aggregate(pdir: str | PathLike, verbose: bool = False) -> None:
    pdir = Path(pdir)
    dump = pdir.parent.stem
    pfout = pdir.parent / f"{dump}_agg_stats.json"

    stats = {"filter": defaultdict(Counter), "writer": Counter()}
    for pfin in tqdm(list(pdir.rglob("*.json")), unit="file"):
        data = json.loads(pfin.read_text(encoding="utf-8"))
        for comp_idx, component in enumerate(data):
            # Remove emojis and such
            full_name = re.sub(r"[^\w\-:]+", "", component["name"])
            full_name = full_name.split("-", 1)[1]
            comp_type, comp_name = full_name.split(":", 1)
            comp_type = comp_type.strip().lower()
            comp_name = comp_name.strip()
            comp_name = f"{comp_name} (#{comp_idx})"

            if comp_type == "filter" and comp_name != "Url-filter":
                stats["filter"][comp_name]["num_input_docs"] += component["stats"]["total"]
                stats["filter"][comp_name]["num_output_docs"] += component["stats"]["forwarded"]
                stats["filter"][comp_name]["num_dropped_docs"] += component["stats"].get("dropped", 0)
            elif comp_type == "writer":
                for item, value in component["stats"].items():
                    if item.endswith(".jsonl.gz"):
                        language = item.split("/")[0]
                        stats["writer"][language] += value

    for comp_name, counter in stats["filter"].items():
        percent_dropped = counter["num_dropped_docs"] / counter["num_input_docs"]
        stats["filter"][comp_name]["percent_dropped_docs"] = f"{percent_dropped:.2%}"

    # Convert defaultdicts and counters to dicts
    stats["filter"] = dict(stats["filter"])
    stats["filter"] = {comp_name: dict(counter) for comp_name, counter in stats["filter"].items()}
    stats["writer"] = sorted(stats["writer"].items(), key=lambda x: x[1], reverse=True)
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
