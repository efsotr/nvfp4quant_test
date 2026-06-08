import json
from datetime import datetime, timezone
from pathlib import Path

ROWS = []


def pytest_addoption(parser):
    parser.addoption(
        "--simulate",
        action="store_true",
        help="Run the simulate-FP4 ScaleSweep MSE path instead of native FP4.",
    )
    parser.addoption(
        "--result-path",
        type=str,
        default=None,
        help="Path for the JSON test result report.",
    )


def pytest_runtest_logreport(report):
    if report.when == "call" or (report.when == "setup" and report.skipped):
        ROWS.append(
            {
                "nodeid": report.nodeid,
                "outcome": report.outcome,
                "duration_s": report.duration,
                "phase": report.when,
                "longrepr": str(report.longrepr) if report.failed else None,
            }
        )


def pytest_sessionfinish(session, exitstatus):
    simulate = session.config.getoption("--simulate")
    result_path = session.config.getoption("--result-path")
    if result_path is None:
        suffix = "simulate" if simulate else "native"
        result_path = f"../result/test_scalesweep_mse_nvfp4_quant_{suffix}_results.json"

    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "simulate": simulate,
        "exitstatus": exitstatus,
        "summary": {
            "passed": sum(row["outcome"] == "passed" for row in ROWS),
            "failed": sum(row["outcome"] == "failed" for row in ROWS),
            "skipped": sum(row["outcome"] == "skipped" for row in ROWS),
        },
        "rows": ROWS,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
