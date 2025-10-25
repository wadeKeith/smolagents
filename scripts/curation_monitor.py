from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

METRICS_DIR = Path("metrics")
CURATION_LOG = METRICS_DIR / "curation_log.jsonl"


def log_curation_event(
    *,
    company_name: str,
    location_hint: str,
    source: str,
    category: str,
    input_chars: int,
    output_chars: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Append a lightweight telemetry event whenever we run the curation model.
    Helps monitor频率/成本，后续可由脚本聚合。
    """
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "ts": time.time(),
        "company": company_name,
        "location_hint": location_hint,
        "source": source,
        "category": category,
        "input_chars": int(input_chars),
        "output_chars": int(output_chars),
    }
    if extra:
        payload.update(extra)
    with CURATION_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def summarize_events(window_seconds: int = 24 * 3600) -> dict[str, Any]:
    """
    Quick aggregation of metrics in the past `window_seconds`.
    """
    if not CURATION_LOG.exists():
        return {"events": 0, "input_chars": 0, "output_chars": 0, "per_company": {}}
    cutoff = time.time() - window_seconds
    events = 0
    input_chars = 0
    output_chars = 0
    per_company: dict[str, dict[str, Any]] = {}
    with CURATION_LOG.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("ts", 0) < cutoff:
                continue
            events += 1
            input_chars += int(record.get("input_chars", 0))
            output_chars += int(record.get("output_chars", 0))
            company = record.get("company", "unknown")
            stats = per_company.setdefault(company, {"events": 0, "input_chars": 0, "output_chars": 0})
            stats["events"] += 1
            stats["input_chars"] += int(record.get("input_chars", 0))
            stats["output_chars"] += int(record.get("output_chars", 0))
    return {
        "window_seconds": window_seconds,
        "events": events,
        "input_chars": input_chars,
        "output_chars": output_chars,
        "per_company": per_company,
    }


def main(argv: Iterable[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="查看整理流程的调用频率与字符量统计。")
    parser.add_argument("--window-hours", type=float, default=24.0, help="聚合的时间窗口（小时）")
    args = parser.parse_args(argv)
    summary = summarize_events(int(args.window_hours * 3600))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
