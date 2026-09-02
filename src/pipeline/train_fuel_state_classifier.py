import argparse
import csv
import json
import math
import os
import pickle
import random
from collections import Counter, defaultdict

SIGNAL_LABELS = {
    "UPWARD_SHIFT",
    "DOWNWARD_SHIFT",
    "GRADUAL_CHANGE",
    "STABLE_JITTER",
    "OSCILLATION_NOISE",
    "UNKNOWN",
    "IMPULSE_NOISE",
}

FEATURE_COLUMNS = [
    "FuelLevel",
    "FuelPct",
    "Speed",
    "MotionSpeedKmh",
    "TimeGapMinutes",
    "DeltaFuel",
    "DeltaPct",
    "AbsDeltaFuel",
    "DeltaOverNoise",
    "RollingStd12",
    "RollingStdPct",
    "DistanceMeters",
    "GpsSpeedKmh",
    "HasGPS",
    "capacity_est",
    "noise_sigma_liters",
    "flat_jitter_threshold",
    "spike_threshold",
    "event_threshold",
    "PrevMedian3",
    "FutureMedian3",
    "FutureMedian5",
    "ReturnToPrevLevel",
    "LocalRange5",
    "LocalRange7",
    "PeakReversalFlag",
    "ValleyReversalFlag",
    "TransientScore",
]

DROP_LABELS = {"UNKNOWN", "INVALID", ""}
MERGE_LABELS = {
    "DROPOUT": "OSCILLATION_NOISE",
    "SPIKE": "OSCILLATION_NOISE",
    "SPIKE_UP": "OSCILLATION_NOISE",
    "SPIKE_DOWN": "OSCILLATION_NOISE",
    "TRANSIENT_NOISE": "OSCILLATION_NOISE",
    "TRANSIENT_UP_NOISE": "OSCILLATION_NOISE",
    "TRANSIENT_DOWN_NOISE": "OSCILLATION_NOISE",
    "TRANSIENT_CLUSTER_NOISE": "OSCILLATION_NOISE",
    "NORMAL": "STABLE_JITTER",
}
TRAIN_LABELS = [label for label in SIGNAL_LABELS if label not in {"UNKNOWN", "IMPULSE_NOISE"}]


def _to_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        number = float(str(value).replace(",", "."))
        if math.isnan(number) or math.isinf(number):
            return default
        return number
    except Exception:
        return default


def normalize_label(label):
    label = str(label or "").strip().upper()
    if label in DROP_LABELS:
        return None
    if label not in TRAIN_LABELS:
        return None
    return label


def read_dataset(path):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = normalize_label(row.get("Label"))
            if label is None:
                continue
            features = [_to_float(row.get(col), 0.0) for col in FEATURE_COLUMNS]
            rows.append(
                {
                    "x": features,
                    "y": label,
                    "vehicle": row.get("VehicleID", ""),
                    "time": row.get("FuelTime", ""),
                }
            )
    return rows


def balance_training_rows(rows, stable_limit=30000, min_rare_target=1500, seed=42):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["y"]].append(row)

    balanced = []
    for label in TRAIN_LABELS:
        label_rows = by_label.get(label, [])
        if not label_rows:
            continue

        if label == "STABLE_JITTER" and len(label_rows) > stable_limit:
            balanced.extend(rng.sample(label_rows, stable_limit))
            continue

        balanced.extend(label_rows)

        if label in {"REFUEL", "DRAIN"} and len(label_rows) < min_rare_target:
            needed = min_rare_target - len(label_rows)
            balanced.extend(rng.choice(label_rows) for _ in range(needed))

    rng.shuffle(balanced)
    return balanced


def split_xy(rows):
    return [row["x"] for row in rows], [row["y"] for row in rows]


def save_confusion_matrix_artifacts(out_dir, split_name, labels, matrix):
    csv_path = os.path.join(out_dir, f"{split_name.lower()}_confusion_matrix.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Actual \\ Predicted", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[int(value) for value in row]])

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return csv_path, None

    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(f"{split_name.title()} Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    threshold = matrix.max() * 0.55 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=9,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    png_path = os.path.join(out_dir, f"{split_name.lower()}_confusion_matrix.png")
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return csv_path, png_path


def _safe_filename_part(value):
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))


def write_counter(path, title, counter):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"\n{title}\n")
        for key, value in counter.most_common():
            handle.write(f"{key}: {value}\n")


def main():
    parser = argparse.ArgumentParser(description="Train baseline fuel-state classifier.")
    parser.add_argument("--data-dir", default=r"D:\THUCTAP_VICOMSAT\data\fuel_label_dataset")
    parser.add_argument("--out-dir", default=r"D:\THUCTAP_VICOMSAT\models\fuel_state_classifier")
    parser.add_argument("--model", choices=["random_forest", "hist_gradient_boosting"], default="random_forest")
    parser.add_argument("--stable-limit", type=int, default=30000)
    parser.add_argument("--rare-target", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: scikit-learn. Install it in the environment used for training, "
            "then run this script again."
        ) from exc

    train_path = os.path.join(args.data_dir, "train.csv")
    val_path = os.path.join(args.data_dir, "val.csv")
    test_path = os.path.join(args.data_dir, "test.csv")

    train_rows_raw = read_dataset(train_path)
    val_rows = read_dataset(val_path)
    test_rows = read_dataset(test_path)
    train_rows = balance_training_rows(
        train_rows_raw,
        stable_limit=args.stable_limit,
        min_rare_target=args.rare_target,
        seed=args.seed,
    )

    x_train, y_train = split_xy(train_rows)
    _, y_train_raw = split_xy(train_rows_raw)
    x_val, y_val = split_xy(val_rows)
    x_test, y_test = split_xy(test_rows)

    if args.model == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=18,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=args.seed,
        )
        model = classifier
    else:
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=300,
                        learning_rate=0.05,
                        max_leaf_nodes=31,
                        l2_regularization=0.1,
                        random_state=args.seed,
                    ),
                ),
            ]
        )

    os.makedirs(args.out_dir, exist_ok=True)
    model.fit(x_train, y_train)

    report_path = os.path.join(args.out_dir, "classification_report.txt")
    metadata_path = os.path.join(args.out_dir, "metadata.json")
    model_path = os.path.join(args.out_dir, "fuel_state_classifier.pkl")
    matrix_artifacts = {}
    per_vehicle_test_artifacts = {}

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(f"Model: {args.model}\n")
        handle.write(f"Features: {FEATURE_COLUMNS}\n")
        handle.write(f"Train raw rows: {len(train_rows_raw)}\n")
        handle.write(f"Train balanced rows: {len(train_rows)}\n")
        handle.write(f"Validation rows: {len(val_rows)}\n")
        handle.write(f"Test rows: {len(test_rows)}\n")
        handle.write("\nTrain raw label counts:\n")
        for label, count in Counter(y_train_raw).most_common():
            handle.write(f"{label}: {count}\n")
        handle.write("\nTrain balanced label counts:\n")
        for label, count in Counter(y_train).most_common():
            handle.write(f"{label}: {count}\n")

        test_predictions = None
        for split_name, x_data, y_data in [("VALIDATION", x_val, y_val), ("TEST", x_test, y_test)]:
            predictions = model.predict(x_data)
            if split_name == "TEST":
                test_predictions = predictions
            handle.write(f"\n\n{split_name} CLASSIFICATION REPORT\n")
            handle.write(classification_report(y_data, predictions, labels=TRAIN_LABELS, zero_division=0))
            handle.write(f"\n{split_name} CONFUSION MATRIX labels={TRAIN_LABELS}\n")
            matrix = confusion_matrix(y_data, predictions, labels=TRAIN_LABELS)
            for row in matrix:
                handle.write(",".join(str(int(value)) for value in row) + "\n")
            csv_path, png_path = save_confusion_matrix_artifacts(args.out_dir, split_name, TRAIN_LABELS, matrix)
            matrix_artifacts[split_name.lower()] = {"csv": csv_path, "png": png_path}

        if test_predictions is not None:
            by_vehicle = defaultdict(list)
            for index, row in enumerate(test_rows):
                by_vehicle[row["vehicle"]].append(index)

            handle.write("\n\nPER-VEHICLE TEST RESULTS\n")
            for vehicle_id in sorted(by_vehicle):
                indices = by_vehicle[vehicle_id]
                y_vehicle = [y_test[index] for index in indices]
                predictions_vehicle = [test_predictions[index] for index in indices]
                matrix = confusion_matrix(y_vehicle, predictions_vehicle, labels=TRAIN_LABELS)
                artifact_name = f"TEST_{_safe_filename_part(vehicle_id)}"
                csv_path, png_path = save_confusion_matrix_artifacts(
                    args.out_dir, artifact_name, TRAIN_LABELS, matrix
                )
                accuracy = sum(
                    actual == predicted for actual, predicted in zip(y_vehicle, predictions_vehicle)
                ) / len(y_vehicle)
                handle.write(f"\n{vehicle_id}: rows={len(y_vehicle)}, accuracy={accuracy:.4f}\n")
                for row in matrix:
                    handle.write(",".join(str(int(value)) for value in row) + "\n")
                per_vehicle_test_artifacts[vehicle_id] = {
                    "rows": len(y_vehicle),
                    "accuracy": accuracy,
                    "csv": csv_path,
                    "png": png_path,
                }

    metadata = {
        "model_type": args.model,
        "feature_columns": FEATURE_COLUMNS,
        "labels": TRAIN_LABELS,
        "merge_labels": MERGE_LABELS,
        "drop_labels": sorted(DROP_LABELS),
        "train_raw_count": len(train_rows_raw),
        "train_balanced_count": len(train_rows),
        "val_count": len(val_rows),
        "test_count": len(test_rows),
        "stable_limit": args.stable_limit,
        "rare_target": args.rare_target,
        "confusion_matrix_artifacts": matrix_artifacts,
        "per_vehicle_test_artifacts": per_vehicle_test_artifacts,
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    with open(model_path, "wb") as handle:
        pickle.dump(model, handle)

    print(f"Saved model: {model_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved metadata: {metadata_path}")
    for split_name, paths in matrix_artifacts.items():
        print(f"Saved {split_name} confusion matrix CSV: {paths['csv']}")
        if paths["png"]:
            print(f"Saved {split_name} confusion matrix PNG: {paths['png']}")
    for vehicle_id, paths in per_vehicle_test_artifacts.items():
        print(f"Saved test metrics for {vehicle_id}: accuracy={paths['accuracy']:.4f}")
        print(f"Saved test confusion matrix CSV: {paths['csv']}")
        if paths["png"]:
            print(f"Saved test confusion matrix PNG: {paths['png']}")


if __name__ == "__main__":
    main()
