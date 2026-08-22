from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
from torch import nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
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
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(self.net(x) + residual)


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


def load_fuel_state_tcn(model_dir: str = "models/fuel_state_tcn"):
    metadata_path = os.path.join(model_dir, "metadata.json")
    model_path = os.path.join(model_dir, "fuel_state_tcn.pt")
    if not os.path.exists(metadata_path) or not os.path.exists(model_path):
        return None, None
    try:
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        model = FuelTCN(
            input_dim=int(metadata.get("input_dim", len(metadata.get("feature_columns", [])))),
            num_classes=len(metadata.get("labels", [])),
            channels=tuple(metadata.get("channels", [64, 64, 64])),
            kernel_size=int(metadata.get("kernel_size", 3)),
            dropout=float(metadata.get("dropout", 0.15)),
        )
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return model, metadata
    except Exception:
        return None, None


def _build_tcn_windows(features: np.ndarray, window_size: int) -> np.ndarray:
    half = window_size // 2
    zero = np.zeros(features.shape[1], dtype=np.float32)
    windows = []
    for idx in range(len(features)):
        rows = []
        for offset in range(-half, half + 1):
            src = idx + offset
            rows.append(features[src] if 0 <= src < len(features) else zero)
        windows.append(np.stack(rows, axis=0))
    return np.transpose(np.stack(windows, axis=0).astype(np.float32), (0, 2, 1))


def predict_tcn_fuel_state(df: pd.DataFrame, model, metadata: dict | None, batch_size: int = 2048) -> pd.DataFrame:
    result = df.copy()
    result["TCN_State"] = "MODEL_NOT_FOUND"
    result["TCN_Confidence"] = np.nan
    if model is None or not metadata:
        return result

    feature_columns = metadata.get("feature_columns", [])
    labels = metadata.get("labels", [])
    if not feature_columns or not labels:
        return result

    for column in feature_columns:
        if column not in result.columns:
            result[column] = 0.0
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)

    mean = np.asarray(metadata.get("mean", [0.0] * len(feature_columns)), dtype=np.float32)
    std = np.asarray(metadata.get("std", [1.0] * len(feature_columns)), dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    window_size = int(metadata.get("window_size", 21))

    predicted = pd.Series("UNKNOWN", index=result.index, dtype=object)
    confidence = pd.Series(np.nan, index=result.index, dtype=float)
    groups = result.groupby("SegmentID", sort=False, dropna=False) if "SegmentID" in result.columns else [(0, result)]

    with torch.no_grad():
        for _, group in groups:
            ordered = group.sort_values("FuelTime", kind="stable") if "FuelTime" in group.columns else group
            features = ordered[feature_columns].to_numpy(dtype=np.float32)
            features = (features - mean) / std
            windows = _build_tcn_windows(features, window_size)
            all_ids = []
            all_conf = []
            for start in range(0, len(windows), batch_size):
                batch = torch.from_numpy(windows[start : start + batch_size])
                probs = torch.softmax(model(batch), dim=1)
                conf, ids = torch.max(probs, dim=1)
                all_ids.extend(ids.cpu().numpy().tolist())
                all_conf.extend(conf.cpu().numpy().tolist())
            predicted.loc[ordered.index] = [labels[int(label_id)] for label_id in all_ids]
            confidence.loc[ordered.index] = all_conf

    result["TCN_State"] = predicted
    result["TCN_Confidence"] = confidence
    return result
