import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.train_fuel_state_classifier import (
    DROP_LABELS,
    FEATURE_COLUMNS,
    MERGE_LABELS,
    TRAIN_LABELS,
    normalize_label,
)


LABEL_TO_ID = {label: idx for idx, label in enumerate(TRAIN_LABELS)}


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


def read_sequence_rows(path):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = normalize_label(row.get("Label"))
            if label is None:
                continue
            rows.append(
                {
                    "vehicle": row.get("VehicleID", ""),
                    "segment": row.get("SegmentID", ""),
                    "time": row.get("FuelTime", ""),
                    "x": [_to_float(row.get(col), 0.0) for col in FEATURE_COLUMNS],
                    "y": label,
                }
            )
    rows.sort(key=lambda item: (item["vehicle"], item["segment"], item["time"]))
    return rows


def build_windows(rows, window_size=15, stable_limit=45000, rare_target=2500, seed=42):
    rng = random.Random(seed)
    half = window_size // 2
    by_group = defaultdict(list)
    for row in rows:
        by_group[(row["vehicle"], row["segment"])].append(row)

    by_label = defaultdict(list)
    feature_count = len(FEATURE_COLUMNS)
    zero = np.zeros(feature_count, dtype=np.float32)

    for group_rows in by_group.values():
        xs = [np.asarray(row["x"], dtype=np.float32) for row in group_rows]
        for idx, row in enumerate(group_rows):
            window = []
            for offset in range(-half, half + 1):
                src = idx + offset
                window.append(xs[src] if 0 <= src < len(xs) else zero)
            by_label[row["y"]].append((np.stack(window, axis=0), LABEL_TO_ID[row["y"]]))

    samples = []
    for label in TRAIN_LABELS:
        label_samples = by_label.get(label, [])
        if not label_samples:
            continue
        if label == "STABLE_JITTER" and len(label_samples) > stable_limit:
            samples.extend(rng.sample(label_samples, stable_limit))
        else:
            samples.extend(label_samples)
        if label in {"REFUEL", "DRAIN", "SPIKE", "TRANSIENT_NOISE"} and len(label_samples) < rare_target:
            samples.extend(rng.choice(label_samples) for _ in range(rare_target - len(label_samples)))

    rng.shuffle(samples)
    return samples, {label: len(by_label.get(label, [])) for label in TRAIN_LABELS}


class FuelWindowDataset(Dataset):
    def __init__(self, samples, mean=None, std=None):
        self.x = np.stack([sample[0] for sample in samples], axis=0).astype(np.float32)
        self.y = np.asarray([sample[1] for sample in samples], dtype=np.int64)
        if mean is None:
            mean = self.x.reshape(-1, self.x.shape[-1]).mean(axis=0)
        if std is None:
            std = self.x.reshape(-1, self.x.shape[-1]).std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.x = (self.x - self.mean) / self.std
        self.x = np.transpose(self.x, (0, 2, 1))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return torch.from_numpy(self.x[index]), torch.tensor(self.y[index], dtype=torch.long)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)


class FuelTCN(nn.Module):
    def __init__(self, input_dim, num_classes, channels=(64, 64, 64), kernel_size=3, dropout=0.15):
        super().__init__()
        blocks = []
        in_channels = input_dim
        for level, out_channels in enumerate(channels):
            blocks.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=2**level,
                    dropout=dropout,
                )
            )
            in_channels = out_channels
        self.tcn = nn.Sequential(*blocks)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        features = self.tcn(x)
        center = features[:, :, features.shape[-1] // 2]
        return self.classifier(center)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    matrix = np.zeros((len(TRAIN_LABELS), len(TRAIN_LABELS)), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            for actual, predicted in zip(y.cpu().numpy(), pred.cpu().numpy()):
                matrix[int(actual), int(predicted)] += 1
    return correct / max(total, 1), matrix


def save_matrix(out_dir, split_name, matrix):
    csv_path = os.path.join(out_dir, f"{split_name}_confusion_matrix.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Actual \\ Predicted", *TRAIN_LABELS])
        for label, row in zip(TRAIN_LABELS, matrix):
            writer.writerow([label, *[int(value) for value in row]])

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return csv_path, None

    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(f"{split_name.title()} TCN Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks(range(len(TRAIN_LABELS)))
    ax.set_yticks(range(len(TRAIN_LABELS)))
    ax.set_xticklabels(TRAIN_LABELS, rotation=35, ha="right")
    ax.set_yticklabels(TRAIN_LABELS)
    threshold = matrix.max() * 0.55 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value > threshold else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    png_path = os.path.join(out_dir, f"{split_name}_confusion_matrix.png")
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return csv_path, png_path


def main():
    parser = argparse.ArgumentParser(description="Train TCN fuel-state sequence classifier.")
    parser.add_argument("--data-dir", default=r"D:\THUCTAP_VICOMSAT\data\fuel_label_dataset")
    parser.add_argument("--out-dir", default=r"D:\THUCTAP_VICOMSAT\models\fuel_state_tcn")
    parser.add_argument("--window-size", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--stable-limit", type=int, default=45000)
    parser.add_argument("--rare-target", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_rows = read_sequence_rows(os.path.join(args.data_dir, "train.csv"))
    val_rows = read_sequence_rows(os.path.join(args.data_dir, "val.csv"))
    test_rows = read_sequence_rows(os.path.join(args.data_dir, "test.csv"))

    train_samples, train_counts = build_windows(
        train_rows,
        window_size=args.window_size,
        stable_limit=args.stable_limit,
        rare_target=args.rare_target,
        seed=args.seed,
    )
    val_samples, val_counts = build_windows(val_rows, window_size=args.window_size, stable_limit=10**9, rare_target=0, seed=args.seed)
    test_samples, test_counts = build_windows(test_rows, window_size=args.window_size, stable_limit=10**9, rare_target=0, seed=args.seed)

    train_dataset = FuelWindowDataset(train_samples)
    val_dataset = FuelWindowDataset(val_samples, mean=train_dataset.mean, std=train_dataset.std)
    test_dataset = FuelWindowDataset(test_samples, mean=train_dataset.mean, std=train_dataset.std)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FuelTCN(input_dim=len(FEATURE_COLUMNS), num_classes=len(TRAIN_LABELS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    label_counts = Counter(label_id for _, label_id in train_samples)
    total = sum(label_counts.values())
    weights = []
    for label in TRAIN_LABELS:
        count = max(label_counts.get(LABEL_TO_ID[label], 0), 1)
        weights.append(total / (len(TRAIN_LABELS) * count))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    os.makedirs(args.out_dir, exist_ok=True)
    best_val = -1.0
    best_path = os.path.join(args.out_dir, "fuel_state_tcn.pt")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * int(y.numel())
        train_loss = running_loss / max(len(train_dataset), 1)
        val_acc, _ = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_accuracy": val_acc})
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    val_acc, val_matrix = evaluate(model, val_loader, device)
    test_acc, test_matrix = evaluate(model, test_loader, device)
    val_csv, val_png = save_matrix(args.out_dir, "validation", val_matrix)
    test_csv, test_png = save_matrix(args.out_dir, "test", test_matrix)

    metadata = {
        "model_type": "tcn",
        "feature_columns": FEATURE_COLUMNS,
        "labels": TRAIN_LABELS,
        "merge_labels": MERGE_LABELS,
        "drop_labels": sorted(DROP_LABELS),
        "window_size": args.window_size,
        "input_dim": len(FEATURE_COLUMNS),
        "channels": [64, 64, 64],
        "kernel_size": 3,
        "dropout": 0.15,
        "mean": train_dataset.mean.tolist(),
        "std": train_dataset.std.tolist(),
        "train_counts": train_counts,
        "val_counts": val_counts,
        "test_counts": test_counts,
        "history": history,
        "best_val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "artifacts": {
            "model": best_path,
            "validation_confusion_matrix_csv": val_csv,
            "validation_confusion_matrix_png": val_png,
            "test_confusion_matrix_csv": test_csv,
            "test_confusion_matrix_png": test_png,
        },
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "classification_report.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"Validation accuracy: {val_acc:.6f}\n")
        handle.write(f"Test accuracy: {test_acc:.6f}\n")
        handle.write(f"Labels: {TRAIN_LABELS}\n")
        handle.write(f"Window size: {args.window_size}\n")
        handle.write("\nTrain raw counts:\n")
        for label, count in train_counts.items():
            handle.write(f"{label}: {count}\n")

    print(f"Saved model: {best_path}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
