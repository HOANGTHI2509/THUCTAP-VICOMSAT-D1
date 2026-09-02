"""Create the slide-ready per-vehicle function matrix for an RF artifact."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np

from src.pipeline import train_fuel_state_classifier as trainer


DEFAULT_EXCLUDED_VEHICLES = {"Car 5"}


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-vehicle RF function matrix.")
    parser.add_argument("--model-dir", default="models/rf_signal_state_causal_v3")
    parser.add_argument("--data-dir", default="data/fuel_label_dataset")
    parser.add_argument("--out-dir", default="models/rf_signal_state_causal_v3")
    parser.add_argument("--include-excluded", action="store_true", help="Also include Car 5 in the main matrix.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    with (model_dir / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    with (model_dir / "fuel_state_classifier.pkl").open("rb") as handle:
        model = pickle.load(handle)

    # ``read_dataset`` follows FEATURE_COLUMNS, so make it exactly match the
    # selected artifact rather than the offline/default trainer features.
    trainer.FEATURE_COLUMNS = list(metadata["feature_columns"])
    labels = list(metadata["labels"])
    rows = trainer.read_dataset(str(Path(args.data_dir) / "test.csv"))

    x_data, y_true = trainer.split_xy(rows)
    y_pred = model.predict(x_data)
    probabilities = model.predict_proba(x_data) if hasattr(model, "predict_proba") else None
    classes = list(getattr(model, "classes_", labels))
    class_index = {label: index for index, label in enumerate(classes)}

    by_vehicle: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_vehicle.setdefault(str(row["vehicle"] or "UNKNOWN_VEHICLE"), []).append(index)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "function_matrix_per_vehicle.csv"
    markdown_path = out_dir / "function_matrix_per_vehicle.md"
    excluded = set() if args.include_excluded else DEFAULT_EXCLUDED_VEHICLES

    result_rows: list[dict[str, object]] = []
    for vehicle_id in sorted(by_vehicle):
        if vehicle_id in excluded:
            continue
        indices = np.asarray(by_vehicle[vehicle_id], dtype=int)
        true_vehicle = np.asarray(y_true, dtype=object)[indices]
        pred_vehicle = np.asarray(y_pred, dtype=object)[indices]
        for label in labels:
            actual = true_vehicle == label
            predicted = pred_vehicle == label
            support = int(actual.sum())
            tp = int((actual & predicted).sum())
            fp = int((~actual & predicted).sum())
            fn = int((actual & ~predicted).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
            confidence = ""
            if probabilities is not None and label in class_index and support:
                confidence = round(float(probabilities[indices[actual], class_index[label]].mean()), 4)
            result_rows.append(
                {
                    "VehicleID": vehicle_id,
                    "Label": label,
                    "Support": support,
                    "Precision": round(precision, 4),
                    "Recall": round(recall, 4),
                    "F1": round(f1, 4),
                    "MeanTrueClassConfidence": confidence,
                }
            )

    fields = ["VehicleID", "Label", "Support", "Precision", "Recall", "F1", "MeanTrueClassConfidence"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result_rows)

    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write("# RF Causal v3 — Function Matrix theo xe\n\n")
        handle.write("Tập test chính: 90H-03494 và 92H-02687. Car 5 được loại khỏi bảng chính do mất tín hiệu cảm biến thường xuyên. ")
        handle.write("Dấu — nghĩa là nhãn không có mẫu thực tế trên xe đó, nên không dùng để kết luận metric riêng theo xe.\n\n")
        handle.write("| Vehicle | Label | Support | Precision | Recall | F1 | Mean confidence |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|\n")
        for row in result_rows:
            confidence = row["MeanTrueClassConfidence"]
            has_support = int(row["Support"]) > 0
            precision_text = f"{float(row['Precision']):.2%}" if has_support else "—"
            recall_text = f"{float(row['Recall']):.2%}" if has_support else "—"
            f1_text = f"{float(row['F1']):.2%}" if has_support else "—"
            confidence_text = "—" if not has_support or confidence == "" else f"{float(confidence):.2%}"
            handle.write(
                f"| {row['VehicleID']} | {row['Label']} | {row['Support']} | "
                f"{precision_text} | {recall_text} | {f1_text} | {confidence_text} |\n"
            )

    print(f"Saved CSV: {csv_path}")
    print(f"Saved Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
