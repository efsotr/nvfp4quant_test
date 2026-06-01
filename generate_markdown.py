import argparse
import json
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("result")
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "benchmark_report.md"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bench-results", nargs="*", type=Path)
    parser.add_argument("--gemm-results", nargs="*", type=Path)
    return parser.parse_args()


def load_json(path):
    return json.loads(path.read_text())


def default_bench_paths(output_dir):
    return sorted(output_dir.glob("bench_*_results.json"))


def default_gemm_paths(output_dir):
    return sorted(output_dir.glob("gemm_*_results.json"))


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def table(headers, rows):
    if not rows:
        return "_No results found._\n"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def bench_rows(paths):
    rows = []
    for path in paths:
        data = load_json(path)
        name = data.get("name", path.stem).removeprefix("triton.")
        layout = data.get("scale_layout", "-")
        for item in data.get("results", []):
            rows.append(
                [
                    name,
                    layout,
                    item.get("bsz"),
                    item.get("dim", data.get("dim")),
                    item.get("latency_ms"),
                    item.get("mse"),
                    item.get("weighted_mse"),
                    item.get("max_abs_error"),
                ]
            )
    return rows


def gemm_rows(paths):
    rows = []
    for path in paths:
        data = load_json(path)
        mode = data.get("mode", data.get("name", path.stem))
        for item in data.get("results", []):
            rows.append(
                [
                    mode,
                    item.get("kernel"),
                    item.get("status", "ok"),
                    item.get("latency_ms"),
                    item.get("mse"),
                    item.get("max_abs_error"),
                    item.get("reason") or item.get("error"),
                ]
            )
    return rows


def environment_rows(paths):
    rows = []
    seen = set()
    for path in paths:
        data = load_json(path)
        env = data.get("environment") or {}
        gpu = env.get("gpu") or {}
        row = (
            env.get("torch"),
            env.get("vllm"),
            env.get("cuda"),
            env.get("gpu_driver"),
            gpu.get("name"),
            gpu.get("sm"),
            gpu.get("sm_count"),
            gpu.get("total_memory_bytes"),
        )
        if row in seen or not any(value is not None for value in row):
            continue
        seen.add(row)
        rows.append(row)
    return rows


def metadata_rows(paths):
    rows = []
    for path in paths:
        data = load_json(path)
        rows.append(
            [
                path.name,
                data.get("name"),
                data.get("mode", "-"),
                data.get("sm", (data.get("environment") or {}).get("gpu", {}).get("sm", "-")),
                data.get("sm_count", (data.get("environment") or {}).get("gpu", {}).get("sm_count", "-")),
                data.get("dim", "-"),
                data.get("weight_distribution", "-"),
                data.get("input_distribution", "-"),
                data.get("channel_square_norm", "-"),
            ]
        )
    return rows


def render(bench_paths, gemm_paths):
    all_paths = [*bench_paths, *gemm_paths]
    lines = [
        "# NVFP4 Benchmark Report",
        "",
        "## Environment",
        "",
        table(
            ["torch", "vllm", "cuda", "gpu_driver", "gpu", "sm", "sm_count", "total_memory_bytes"],
            environment_rows(all_paths),
        ),
        "## Result Files",
        "",
        table(
            [
                "file",
                "name",
                "mode",
                "sm",
                "sm_count",
                "dim",
                "weight_distribution",
                "input_distribution",
                "channel_square_norm",
            ],
            metadata_rows(all_paths),
        ),
        "## GEMM Results",
        "",
        table(
            ["mode", "kernel", "status", "latency_ms", "mse", "max_abs_error", "note"],
            gemm_rows(gemm_paths),
        ),
        "## Quantization Bench Results",
        "",
        table(
            ["kernel", "layout", "bsz", "dim", "latency_ms", "mse", "weighted_mse", "max_abs_error"],
            bench_rows(bench_paths),
        ),
    ]
    return "\n".join(lines).rstrip() + "\n"


def main():
    args = parse_args()
    bench_paths = args.bench_results if args.bench_results is not None else default_bench_paths(args.output_dir)
    gemm_paths = args.gemm_results if args.gemm_results is not None else default_gemm_paths(args.output_dir)
    output = args.output or args.output_dir / DEFAULT_OUTPUT.name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(bench_paths, gemm_paths))
    print(f"saved markdown report to {output}")


if __name__ == "__main__":
    main()
