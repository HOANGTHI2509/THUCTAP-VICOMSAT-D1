import argparse
import csv
import os


def read_confusion_matrix(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        labels = header[1:]
        matrix = []
        row_labels = []
        for row in reader:
            row_labels.append(row[0])
            matrix.append([int(float(value)) for value in row[1:]])
    return labels, row_labels, matrix


def compute_metrics(labels, matrix):
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(len(labels)))
    rows = []
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        actual = sum(matrix[i])
        predicted = sum(matrix[row][i] for row in range(len(labels)))
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "label": label,
                "support": actual,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    macro_f1 = sum(row["f1"] for row in rows) / len(rows) if rows else 0.0
    weighted_f1 = sum(row["f1"] * row["support"] for row in rows) / total if total else 0.0
    accuracy = correct / total if total else 0.0
    return rows, {"accuracy": accuracy, "macro_f1": macro_f1, "weighted_f1": weighted_f1, "total": total}


def fmt(value):
    return f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser(description="Compare RF and TCN fuel-state confusion matrices.")
    parser.add_argument("--rf", default=r"D:\THUCTAP_VICOMSAT\models\fuel_state_classifier\test_confusion_matrix.csv")
    parser.add_argument("--tcn", default=r"D:\THUCTAP_VICOMSAT\models\fuel_state_tcn\test_confusion_matrix.csv")
    parser.add_argument("--out-dir", default=r"D:\THUCTAP_VICOMSAT\models\fuel_state_comparison")
    args = parser.parse_args()

    rf_labels, _, rf_matrix = read_confusion_matrix(args.rf)
    tcn_labels, _, tcn_matrix = read_confusion_matrix(args.tcn)
    if rf_labels != tcn_labels:
        raise SystemExit(f"Label mismatch: RF={rf_labels}, TCN={tcn_labels}")

    rf_rows, rf_summary = compute_metrics(rf_labels, rf_matrix)
    tcn_rows, tcn_summary = compute_metrics(tcn_labels, tcn_matrix)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "rf_vs_tcn_metrics.csv")
    md_path = os.path.join(args.out_dir, "rf_vs_tcn_metrics.md")

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Label",
                "Support",
                "RF Precision",
                "RF Recall",
                "RF F1",
                "TCN Precision",
                "TCN Recall",
                "TCN F1",
                "F1 Delta (TCN-RF)",
            ]
        )
        for rf_row, tcn_row in zip(rf_rows, tcn_rows):
            writer.writerow(
                [
                    rf_row["label"],
                    rf_row["support"],
                    fmt(rf_row["precision"]),
                    fmt(rf_row["recall"]),
                    fmt(rf_row["f1"]),
                    fmt(tcn_row["precision"]),
                    fmt(tcn_row["recall"]),
                    fmt(tcn_row["f1"]),
                    fmt(tcn_row["f1"] - rf_row["f1"]),
                ]
            )
        writer.writerow([])
        writer.writerow(["Summary", "RF", "TCN", "Delta (TCN-RF)"])
        for key in ["accuracy", "macro_f1", "weighted_f1"]:
            writer.writerow([key, fmt(rf_summary[key]), fmt(tcn_summary[key]), fmt(tcn_summary[key] - rf_summary[key])])

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("# RF vs TCN Fuel State Classifier\n\n")
        handle.write("| Label | Support | RF Precision | RF Recall | RF F1 | TCN Precision | TCN Recall | TCN F1 | F1 Delta |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for rf_row, tcn_row in zip(rf_rows, tcn_rows):
            handle.write(
                f"| {rf_row['label']} | {rf_row['support']} | {fmt(rf_row['precision'])} | {fmt(rf_row['recall'])} | "
                f"{fmt(rf_row['f1'])} | {fmt(tcn_row['precision'])} | {fmt(tcn_row['recall'])} | "
                f"{fmt(tcn_row['f1'])} | {fmt(tcn_row['f1'] - rf_row['f1'])} |\n"
            )
        handle.write("\n")
        handle.write("| Metric | RF | TCN | Delta |\n")
        handle.write("|---|---:|---:|---:|\n")
        for key in ["accuracy", "macro_f1", "weighted_f1"]:
            handle.write(f"| {key} | {fmt(rf_summary[key])} | {fmt(tcn_summary[key])} | {fmt(tcn_summary[key] - rf_summary[key])} |\n")

    print(f"Saved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")
    print(f"RF accuracy={rf_summary['accuracy']:.4f}, TCN accuracy={tcn_summary['accuracy']:.4f}")


if __name__ == "__main__":
    main()
