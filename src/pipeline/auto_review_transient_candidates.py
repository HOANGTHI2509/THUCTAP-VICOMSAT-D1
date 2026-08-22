import argparse
import csv
import math
import os
from collections import Counter


REAL_EVENT_LABELS = {"REFUEL", "DRAIN"}


def _as_float(value, default=0.0):
    try:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return default
        return number
    except Exception:
        return default


def _current_label_set(value):
    return {item.strip() for item in str(value or "").split(",") if item.strip()}


def classify_candidate(row, min_return, min_amplitude, min_duration_points, max_duration_points, max_duration_minutes):
    label = str(row.get("suggested_label", ""))
    if label not in {"TRANSIENT_UP_NOISE", "TRANSIENT_DOWN_NOISE"}:
        return None

    current_labels = _current_label_set(row.get("current_labels"))
    if current_labels & REAL_EVENT_LABELS:
        return None

    amplitude = _as_float(row.get("amplitude"))
    return_ratio = _as_float(row.get("return_ratio"))
    duration_points = int(_as_float(row.get("duration_points"), 999))
    duration_minutes = _as_float(row.get("duration_minutes"), 9999.0)

    if amplitude < min_amplitude:
        return None
    if return_ratio < min_return:
        return None
    if duration_points < min_duration_points:
        return None
    if duration_points > max_duration_points:
        return None
    if duration_minutes > max_duration_minutes:
        return None

    return "TRANSIENT_NOISE"


def auto_review(input_path, output_path, min_return, min_amplitude, min_duration_points, max_duration_points, max_duration_minutes):
    with open(input_path, encoding="utf-8-sig", newline="") as handle:
        candidates = list(csv.DictReader(handle))

    rows = []
    for event_id, row in enumerate(candidates):
        final_label = classify_candidate(
            row,
            min_return=min_return,
            min_amplitude=min_amplitude,
            min_duration_points=min_duration_points,
            max_duration_points=max_duration_points,
            max_duration_minutes=max_duration_minutes,
        )
        if not final_label:
            continue
        rows.append(
            {
                "event_id": event_id,
                "decision": "ACCEPT",
                "final_label": final_label,
                "note": "auto_high_confidence_transient",
            }
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event_id", "decision", "final_label", "note"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved auto review: {output_path}")
    print(f"Accepted transient events: {len(rows):,} / {len(candidates):,}")
    for label, count in Counter(row["final_label"] for row in rows).most_common():
        print(f"{label}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Auto-review high-confidence transient event candidates.")
    parser.add_argument("--input", default=r"D:\THUCTAP_VICOMSAT\data\fuel_label_dataset\transient_event_candidates.csv")
    parser.add_argument("--output", default=r"D:\THUCTAP_VICOMSAT\data\fuel_label_dataset\transient_event_review.csv")
    parser.add_argument("--min-return", type=float, default=0.72)
    parser.add_argument("--min-amplitude", type=float, default=3.0)
    parser.add_argument("--min-duration-points", type=int, default=3)
    parser.add_argument("--max-duration-points", type=int, default=6)
    parser.add_argument("--max-duration-minutes", type=float, default=45.0)
    args = parser.parse_args()
    auto_review(
        args.input,
        args.output,
        min_return=args.min_return,
        min_amplitude=args.min_amplitude,
        min_duration_points=args.min_duration_points,
        max_duration_points=args.max_duration_points,
        max_duration_minutes=args.max_duration_minutes,
    )


if __name__ == "__main__":
    main()
