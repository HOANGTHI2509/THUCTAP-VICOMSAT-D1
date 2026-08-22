import argparse
import csv
import os


VALID_FINAL_LABELS = {
    "TRANSIENT_UP_NOISE",
    "TRANSIENT_DOWN_NOISE",
    "TRANSIENT_CLUSTER_NOISE",
    "TRANSIENT_NOISE",
    "SLOSHING_NOISE",
    "SPIKE",
    "REFUEL",
    "DRAIN",
    "UNKNOWN",
}

TRANSIENT_CHILD_LABELS = {
    "TRANSIENT_UP_NOISE",
    "TRANSIENT_DOWN_NOISE",
    "TRANSIENT_CLUSTER_NOISE",
}


def _parse_source_rows(value):
    rows = []
    for item in str(value or "").split(";"):
        item = item.strip()
        if not item:
            continue
        try:
            rows.append(int(item))
        except ValueError:
            pass
    return rows


def _read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def apply_reviews(data_dir, candidate_path, review_path):
    points_path = os.path.join(data_dir, "all_labeled_points.csv")
    if not os.path.exists(points_path):
        raise FileNotFoundError(points_path)
    if not os.path.exists(candidate_path):
        raise FileNotFoundError(candidate_path)
    if not os.path.exists(review_path):
        raise FileNotFoundError(review_path)

    points = _read_csv(points_path)
    candidates = _read_csv(candidate_path)
    reviews = _read_csv(review_path)
    fieldnames = list(points[0].keys()) if points else []

    override_by_row = {}
    for review in reviews:
        decision = str(review.get("decision", "")).upper()
        if decision not in {"ACCEPT", "REFUEL", "DRAIN"}:
            continue
        try:
            event_id = int(review.get("event_id", ""))
        except ValueError:
            continue
        if event_id < 0 or event_id >= len(candidates):
            continue

        final_label = str(review.get("final_label", "")).strip()
        if final_label == "NOT_TRANSIENT" or final_label not in VALID_FINAL_LABELS:
            continue
        for row_index in _parse_source_rows(candidates[event_id].get("source_rows")):
            override_by_row[row_index] = final_label

    changed = 0
    for row_index, row in enumerate(points):
        new_label = override_by_row.get(row_index)
        if new_label and row.get("Label") != new_label:
            row["Label"] = new_label
            changed += 1
        elif row.get("Label") in TRANSIENT_CHILD_LABELS:
            row["Label"] = "TRANSIENT_NOISE"
            changed += 1

    _write_csv(points_path, points, fieldnames)
    for split in ["train", "val", "test"]:
        split_rows = [row for row in points if str(row.get("Split")) == split]
        _write_csv(os.path.join(data_dir, f"{split}.csv"), split_rows, fieldnames)

    print(f"Applied review labels to rows: {len(override_by_row):,}")
    print(f"Changed labels: {changed:,}")
    print("Updated all_labeled_points.csv, train.csv, val.csv, test.csv")


def main():
    parser = argparse.ArgumentParser(description="Apply transient event review labels to dataset CSVs.")
    parser.add_argument("--data-dir", default=r"D:\THUCTAP_VICOMSAT\data\fuel_label_dataset")
    parser.add_argument("--candidates", default=r"D:\THUCTAP_VICOMSAT\data\fuel_label_dataset\transient_event_candidates.csv")
    parser.add_argument("--review", default=r"D:\THUCTAP_VICOMSAT\data\fuel_label_dataset\transient_event_review.csv")
    args = parser.parse_args()
    apply_reviews(args.data_dir, args.candidates, args.review)


if __name__ == "__main__":
    main()
