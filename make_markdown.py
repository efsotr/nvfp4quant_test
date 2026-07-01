#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

DEFAULT_INPUT = Path("result")
METHOD_LABELS = {
    "vllm": "vllm default ops",
    "ScaleSweep_MSE": "scalesweep MSE",
    "ScaleSweep": "scalesweep",
}
METHOD_ORDER = ["vllm default ops", "scalesweep MSE", "scalesweep"]
METRICS = [
    ("latency_ms", "Latency (ms)"),
    ("mse", "MSE"),
    ("weighted_mse", "Weighted MSE"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert bench.py JSON result files into a Markdown report."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[DEFAULT_INPUT],
        help="JSON files or directories containing JSON files. Defaults to result/.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write Markdown to this file. If omitted, print to stdout.",
    )
    parser.add_argument(
        "--title",
        default="NVFP4 Quantization Benchmark Results",
        help="Markdown report title.",
    )
    return parser.parse_args()


def iter_json_files(inputs):
    paths = []
    for item in inputs:
        if item.is_dir():
            paths.extend(sorted(item.glob("*.json")))
        elif item.is_file():
            paths.append(item)
        else:
            raise FileNotFoundError(f"input path does not exist: {item}")

    seen = set()
    unique_paths = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(path)
    if not unique_paths:
        raise FileNotFoundError("no JSON files found")
    return unique_paths


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    if "results" not in data or not isinstance(data["results"], list):
        raise ValueError(f"{path}: expected a top-level 'results' list")
    return data


def fmt_value(value):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == 0:
            return "0"
        abs_value = abs(value)
        if abs_value < 1e-3 or abs_value >= 1e4:
            return f"{value:.4e}"
        return f"{value:.6g}"
    return str(value).replace("\n", " ")


def escape_md(value):
    return fmt_value(value).replace("|", "\\|")


def make_table(headers, rows):
    lines = ["| " + " | ".join(escape_md(header) for header in headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(escape_md(value) for value in row) + " |")
    return lines


def raw_benchmark_name(data):
    return str(data.get("name", "unknown")).removeprefix("triton.")


def method_label(data):
    raw_name = raw_benchmark_name(data)
    return METHOD_LABELS.get(raw_name, raw_name)


def method_sort_key(item):
    label = method_label(item[1])
    try:
        return METHOD_ORDER.index(label), label
    except ValueError:
        return len(METHOD_ORDER), label


def compact_env(environment):
    if not isinstance(environment, dict) or not environment:
        return "-"
    preferred_keys = [
        "gpu_name",
        "device_name",
        "torch_version",
        "cuda_version",
        "triton_version",
        "python_version",
    ]
    parts = []
    for key in preferred_keys:
        if key in environment:
            parts.append(f"{key}={fmt_value(environment[key])}")
    if parts:
        return ", ".join(parts)
    return ", ".join(f"{key}={fmt_value(value)}" for key, value in sorted(environment.items()))


def collect_bsz(data_items):
    values = set()
    for _, data in data_items:
        for row in data.get("results", []):
            if "bsz" in row:
                values.add(int(row["bsz"]))
    return sorted(values)


def metric_table_rows(data_items, bsz_values, metric):
    rows = []
    for _, data in sorted(data_items, key=method_sort_key):
        result_by_bsz = {
            int(row["bsz"]): row for row in data.get("results", []) if "bsz" in row
        }
        rows.append(
            [method_label(data)]
            + [result_by_bsz.get(bsz, {}).get(metric) for bsz in bsz_values]
        )
    return rows


def common_value(data_items, key):
    values = [data.get(key) for _, data in data_items if data.get(key) is not None]
    if not values:
        return None
    first = values[0]
    if all(value == first for value in values):
        return first
    return ", ".join(fmt_value(value) for value in values)


def render_metadata(data_items):
    lines = []
    lines.append(f"- dim: {fmt_value(common_value(data_items, 'dim'))}")
    lines.append(f"- sm_count: {fmt_value(common_value(data_items, 'sm_count'))}")
    lines.append(f"- scale_layout: {fmt_value(common_value(data_items, 'scale_layout'))}")
    lines.append(
        f"- weight_distribution: {fmt_value(common_value(data_items, 'weight_distribution'))}"
    )

    env_values = [compact_env(data.get("environment")) for _, data in data_items]
    unique_env_values = list(dict.fromkeys(env_values))
    if len(unique_env_values) == 1:
        lines.append(f"- environment: {unique_env_values[0]}")
    else:
        lines.append("- environment: mixed")
    return lines


def render_markdown(data_items, title):
    data_items = sorted(data_items, key=method_sort_key)
    bsz_values = collect_bsz(data_items)
    bsz_headers = [str(bsz) for bsz in bsz_values]

    lines = [f"# {title}", ""]
    lines.extend(render_metadata(data_items))
    lines.append("")

    headers = ["method"] + bsz_headers
    for metric, title_text in METRICS:
        lines.append(f"## {title_text}")
        lines.append("")
        if bsz_values:
            lines.extend(make_table(headers, metric_table_rows(data_items, bsz_values, metric)))
        else:
            lines.append("No result rows found.")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main():
    args = parse_args()
    paths = iter_json_files(args.inputs)
    data_items = [(path, load_json(path)) for path in paths]
    markdown = render_markdown(data_items, args.title)

    if args.output is None:
        print(markdown, end="")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"saved markdown to {args.output}")


if __name__ == "__main__":
    main()
