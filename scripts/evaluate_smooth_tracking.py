"""Reproducible KPI report for the Topic 1 Smooth-Tracking filter."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.filters.smooth_tracking import AISmoothTrackingFilter


DEFAULT_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "golden_fuel_segments.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "evaluation"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "fuel_state_classifier"


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _file_hash(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _detrended_std(values: List[float]) -> float:
    """Estimate local noise without treating a real linear trend as noise."""
    if len(values) < 3:
        return 0.0
    mean_index = (len(values) - 1) / 2.0
    mean_value = statistics.fmean(values)
    denominator = sum((index - mean_index) ** 2 for index in range(len(values)))
    slope = (
        sum(
            (index - mean_index) * (value - mean_value)
            for index, value in enumerate(values)
        )
        / denominator
        if denominator > 0.0
        else 0.0
    )
    residuals = [
        value - (mean_value + slope * (index - mean_index))
        for index, value in enumerate(values)
    ]
    return statistics.pstdev(residuals)


def _run_segment(
    segment: Dict[str, Any],
    model_dir: str,
    measure_latency: bool,
) -> tuple[List[Dict[str, Any]], List[float]]:
    engine = AISmoothTrackingFilter(model_dir=model_dir)
    latitudes = segment.get("latitudes", [None] * len(segment["timestamps"]))
    longitudes = segment.get("longitudes", [None] * len(segment["timestamps"]))
    outputs: List[Dict[str, Any]] = []
    latencies: List[float] = []
    for timestamp, raw, speed, latitude, longitude in zip(
        segment["timestamps"],
        segment["raw_liters"],
        segment["speed_kmh"],
        latitudes,
        longitudes,
    ):
        started = time.perf_counter()
        outputs.append(
            engine.process_point(
                vehicle_id=segment["id"],
                timestamp=timestamp,
                raw_fuel=raw,
                speed=speed,
                capacity_est=segment["capacity_est_liters"],
                lat=latitude,
                lng=longitude,
            )
        )
        latencies.append(
            (time.perf_counter() - started) * 1000.0 if measure_latency else 0.0
        )
    return outputs, latencies


def _behavior_checks(
    segment: Dict[str, Any],
    outputs: List[Dict[str, Any]],
) -> Dict[str, bool]:
    checks = segment.get("checks", {})
    clean = [float(item["clean_fuel"]) for item in outputs]
    raw = [float(value) for value in segment["raw_liters"]]
    results: Dict[str, bool] = {}

    if "max_clean_span" in checks:
        results["max_clean_span"] = max(clean) - min(clean) <= checks["max_clean_span"]
    if "min_clean_liters" in checks:
        results["min_clean_liters"] = min(clean) >= checks["min_clean_liters"]
    if "min_valley_attenuation_liters" in checks:
        results["min_valley_attenuation_liters"] = (
            min(clean) - min(raw) >= checks["min_valley_attenuation_liters"]
        )
    if "min_total_clean_drop" in checks:
        results["min_total_clean_drop"] = (
            clean[0] - clean[-1] >= checks["min_total_clean_drop"]
        )
    if "max_upward_step" in checks:
        upward = max((next_value - value for value, next_value in zip(clean, clean[1:])), default=0.0)
        results["max_upward_step"] = upward <= checks["max_upward_step"]
    if "max_final_lag_liters" in checks:
        results["max_final_lag_liters"] = (
            clean[-1] - raw[-1] <= checks["max_final_lag_liters"]
        )
    if "max_step_size" in checks:
        step = max((abs(next_value - value) for value, next_value in zip(clean, clean[1:])), default=0.0)
        results["max_step_size"] = step <= checks["max_step_size"]
    if "zero_dropout_indexes" in checks:
        results["zero_dropout_indexes"] = all(
            outputs[index]["quality_flag"] == "ZERO_DROPOUT_HELD"
            and clean[index] == clean[index - 1]
            for index in checks["zero_dropout_indexes"]
        )
    if "required_motion_states" in checks:
        states = {item["motion_state"] for item in outputs}
        results["required_motion_states"] = set(checks["required_motion_states"]).issubset(states)
    return results


def evaluate_segments(
    segments: Iterable[Dict[str, Any]],
    model_dir: str,
    generated_at: Optional[str] = None,
    measure_latency: bool = True,
) -> Dict[str, Any]:
    """Evaluate already-loaded fixtures without mutating golden expectations."""
    segments = list(segments)
    fixture_payload = json.dumps(
        segments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    fixture_hash = hashlib.sha256(fixture_payload).hexdigest()
    config_probe = AISmoothTrackingFilter(model_dir=model_dir)
    config_payload = json.dumps(asdict(config_probe.config), sort_keys=True).encode("utf-8")
    config_hash = hashlib.sha256(config_payload).hexdigest()
    model_path = Path(model_dir) / "fuel_state_classifier.pkl" if model_dir else Path()
    metadata_path = Path(model_dir) / "metadata.json" if model_dir else Path()
    model_hash = _file_hash(model_path) if model_dir else None
    metadata = {}
    if model_dir and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    rows: List[Dict[str, Any]] = []
    all_latencies: List[float] = []
    passed_checks = 0
    total_checks = 0
    total_points = 0
    for segment in segments:
        outputs, latencies = _run_segment(segment, model_dir, measure_latency)
        all_latencies.extend(latencies)
        total_points += len(outputs)
        raw = [float(value) for value in segment["raw_liters"]]
        clean = [float(item["clean_fuel"]) for item in outputs]
        valid_pairs = [
            (raw_value, clean_value)
            for raw_value, clean_value in zip(raw, clean)
            if raw_value > 0.0 and math.isfinite(raw_value)
        ]
        valid_raw = [pair[0] for pair in valid_pairs]
        valid_clean = [pair[1] for pair in valid_pairs]
        raw_std = statistics.pstdev(valid_raw) if len(valid_raw) >= 2 else 0.0
        clean_std = statistics.pstdev(clean) if len(clean) >= 2 else 0.0
        raw_detrended_std = _detrended_std(valid_raw)
        clean_detrended_std = _detrended_std(valid_clean)
        noise_reduction = (
            100.0 * (1.0 - clean_detrended_std / raw_detrended_std)
            if raw_detrended_std > 0.0
            else 0.0
        )
        behavior = _behavior_checks(segment, outputs)
        passed = sum(behavior.values())
        passed_checks += passed
        total_checks += len(behavior)
        expected = segment.get("expected_production_clean_liters", clean)
        regression_mae = statistics.fmean(
            abs(actual - target) for actual, target in zip(clean, expected)
        )
        moving_upward_steps = sum(
            current["motion_state"] == "MOVING"
            and current["clean_fuel"] > previous["clean_fuel"] + 0.01
            for previous, current in zip(outputs, outputs[1:])
        )
        raw_range = max(valid_raw) - min(valid_raw) if valid_raw else 0.0
        clean_range = max(clean) - min(clean) if clean else 0.0
        stable_mae = (
            statistics.fmean(abs(c - r) for c, r in zip(clean, raw) if r > 0.0)
            if raw_range <= 1.0
            else None
        )
        total_raw_drop = raw[0] - raw[-1]
        rows.append(
            {
                "segment_id": segment["id"],
                "review_status": segment["review_status"],
                "points": len(outputs),
                "raw_std_liters": round(raw_std, 6),
                "clean_std_liters": round(clean_std, 6),
                "raw_detrended_std_liters": round(raw_detrended_std, 6),
                "clean_detrended_std_liters": round(clean_detrended_std, 6),
                "noise_reduction_percent": round(noise_reduction, 3),
                "raw_range_liters": round(raw_range, 3),
                "clean_range_liters": round(clean_range, 3),
                "spike_attenuation_liters": round(max(0.0, raw_range - clean_range), 3),
                "stable_level_mae_liters": round(stable_mae, 6) if stable_mae is not None else None,
                "final_downtrend_lag_liters": (
                    round(max(0.0, clean[-1] - raw[-1]), 3)
                    if total_raw_drop >= 2.0
                    else None
                ),
                "moving_upward_steps": moving_upward_steps,
                "regression_mae_liters": round(regression_mae, 6),
                "behavior_checks_passed": passed,
                "behavior_checks_total": len(behavior),
                "behavior_checks": behavior,
            }
        )

    total_latency_ms = sum(all_latencies)
    summary = {
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "fixture_segments": len(segments),
        "fixture_points": total_points,
        "fixture_version": f"sha256:{fixture_hash[:12]}",
        "approved_segments": sum(item["review_status"] == "approved" for item in segments),
        "pending_review_segments": sum(item["review_status"] != "approved" for item in segments),
        "golden_checks_passed": passed_checks,
        "golden_checks_total": total_checks,
        "latency_ms_p50": round(_percentile(all_latencies, 0.50), 6),
        "latency_ms_p95": round(_percentile(all_latencies, 0.95), 6),
        "latency_ms_p99": round(_percentile(all_latencies, 0.99), 6),
        "throughput_points_per_second": (
            round(total_points / (total_latency_ms / 1000.0), 3)
            if total_latency_ms > 0.0
            else 0.0
        ),
        "model_type": metadata.get("model_type", "fallback_rules"),
        "model_version": model_hash[:12] if model_hash else "fallback-rules",
        "config_version": f"sha256:{config_hash[:12]}",
        "segments": rows,
    }
    return summary


def write_report(metrics: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows = metrics["segments"]
    csv_fields = [key for key in rows[0] if key != "behavior_checks"] if rows else []
    with (output_dir / "segment_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in csv_fields} for row in rows)

    lines = [
        "# Smooth-Tracking KPI report",
        "",
        f"- Generated: {metrics['generated_at']}",
        f"- Model: {metrics['model_type']} ({metrics['model_version']})",
        f"- Config: {metrics['config_version']}",
        f"- Golden fixture: {metrics['fixture_version']}",
        f"- Golden segments: {metrics['fixture_segments']}",
        f"- Pending domain review: {metrics['pending_review_segments']}",
        f"- Behavior checks: {metrics['golden_checks_passed']}/{metrics['golden_checks_total']}",
        f"- Latency P50/P95/P99: {metrics['latency_ms_p50']}/{metrics['latency_ms_p95']}/{metrics['latency_ms_p99']} ms",
        f"- Throughput: {metrics['throughput_points_per_second']} points/s",
        "",
        "`Noise reduction` is calculated from detrended standard deviation so a",
        "real linear consumption trend is not counted as noise. `Regression MAE`",
        "compares this run with the locked expected production curve.",
        "",
        "| Segment | Review | Noise reduction | Regression MAE | Checks |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['segment_id']} | {row['review_status']} | "
            f"{row['noise_reduction_percent']}% | {row['regression_mae_liters']} L | "
            f"{row['behavior_checks_passed']}/{row['behavior_checks_total']} |"
        )
    failed_checks = [
        (row["segment_id"], name)
        for row in rows
        for name, passed in row["behavior_checks"].items()
        if not passed
    ]
    lines.extend(["", "## Failed behavior checks", ""])
    if failed_checks:
        lines.extend(
            f"- `{segment_id}`: `{check_name}`"
            for segment_id, check_name in failed_checks
        )
    else:
        lines.append("None.")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    segments = json.loads(args.fixture.read_text(encoding="utf-8"))
    metrics = evaluate_segments(segments, model_dir=args.model_dir)
    write_report(metrics, args.output_dir)
    print(
        f"Wrote {args.output_dir} | checks "
        f"{metrics['golden_checks_passed']}/{metrics['golden_checks_total']}"
    )


if __name__ == "__main__":
    main()
