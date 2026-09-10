import json
from pathlib import Path

from scripts.evaluate_smooth_tracking import evaluate_segments, write_report


FIXTURE = Path("tests/fixtures/golden_fuel_segments.json")


def test_kpi_evaluation_is_reproducible_without_timing_noise(tmp_path):
    segments = json.loads(FIXTURE.read_text(encoding="utf-8"))
    arguments = {
        "segments": segments,
        "model_dir": "models/fuel_state_classifier",
        "generated_at": "2026-09-10T00:00:00+00:00",
        "measure_latency": False,
    }
    first = evaluate_segments(**arguments)
    second = evaluate_segments(**arguments)
    assert first == second
    assert first["fixture_segments"] == len(segments)
    assert 0 <= first["golden_checks_passed"] <= first["golden_checks_total"]
    assert first["golden_checks_total"] > 0
    assert first["fixture_version"].startswith("sha256:")
    assert first["config_version"].startswith("sha256:")
    assert first["model_version"] != "fallback-rules"

    write_report(first, tmp_path)
    assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8")) == first
    assert (tmp_path / "segment_metrics.csv").is_file()
    assert "Smooth-Tracking KPI report" in (tmp_path / "report.md").read_text(encoding="utf-8")
