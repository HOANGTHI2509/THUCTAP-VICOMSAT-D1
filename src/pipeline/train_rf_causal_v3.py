"""Train RF Signal-State Causal v3 with present/past-only features."""

from __future__ import annotations

import sys

from src.pipeline import train_fuel_state_classifier as trainer


NON_CAUSAL_FEATURES = {
    "FutureMedian3",
    "FutureMedian5",
    "ReturnToPrevLevel",
    "LocalRange5",
    "LocalRange7",
    "PeakReversalFlag",
    "ValleyReversalFlag",
    "TransientScore",
}
trainer.FEATURE_COLUMNS = [name for name in trainer.FEATURE_COLUMNS if name not in NON_CAUSAL_FEATURES]


def main() -> None:
    if "--out-dir" not in sys.argv:
        sys.argv.extend(["--out-dir", "models/rf_signal_state_causal_v3"])
    trainer.main()


if __name__ == "__main__":
    main()
