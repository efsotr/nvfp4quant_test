#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


METRICS = [
    "prompt_level_strict_acc",
    "inst_level_strict_acc",
    "prompt_level_loose_acc",
    "inst_level_loose_acc",
]


def find_result_file(name_or_path: str, result_dir: Path) -> Path:
    path = Path(name_or_path)
    if path.is_file():
        return path

    candidates = sorted(
        (result_dir / name_or_path).rglob("results_*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Cannot find results_*.json for {name_or_path!r} under {result_dir}"
    )


def load_ifeval_metrics(result_file: Path) -> dict[str, float]:
    with result_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        ifeval = data["results"]["ifeval"]
    except KeyError as exc:
        raise KeyError(f"{result_file} does not contain results.ifeval") from exc

    values = {}
    for metric in METRICS:
        key = f"{metric},none"
        if key not in ifeval:
            raise KeyError(f"{result_file} does not contain {key}")
        values[metric] = ifeval[key]
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the four IFEval metrics from an lm_eval results JSON."
    )
    parser.add_argument("name", help="Model/result name under result/, or a JSON path")
    parser.add_argument(
        "--result-dir",
        default="result",
        type=Path,
        help="Directory containing lm_eval outputs (default: result)",
    )
    args = parser.parse_args()

    try:
        result_file = find_result_file(args.name, args.result_dir)
        metrics = load_ifeval_metrics(result_file)

        print(args.name)
        print(f"result_file: {result_file}")
        for metric in METRICS:
            print(f"{metric}: {metrics[metric]:.6f}")
    except:
        print(f"Error processing {args.name!r}")


if __name__ == "__main__":
    main()
