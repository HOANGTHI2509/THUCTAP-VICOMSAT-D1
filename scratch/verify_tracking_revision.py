"""Read-only replay of the dashboard vehicles and existing regression segments."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.service.state_manager import StreamingStateManager
from src.core.filters.ai_state_filter import load_fuel_state_classifier, filter_with_ai_state
from src.core.filters.ai_enhanced_adaptive_realtime import filter_ai_enhanced_adaptive_realtime


def main():
    for path in Path("artifacts/baseline_results").glob("case[1234]*trace.csv"):
        base = pd.read_csv(path)
        vehicle = path.stem.split("_")[-2]
        raw = pd.read_csv(f"TienXuLy/{vehicle}_processed.csv")
        times = pd.to_datetime(raw.FuelTime)
        bt = pd.to_datetime(base.fuel_time)
        raw = raw[(times >= bt.min()) & (times <= bt.max())]
        manager, trace = StreamingStateManager(), []
        for _, row in raw.iterrows():
            manager.process_point(vehicle_id=vehicle, fuel_time=pd.Timestamp(row.FuelTime),
                fuel_level=row.FuelLevel, speed=row.Speed,
                capacity_est=float(base.capacity_est.iloc[0]),
                noise_sigma_liters=float(base.flat_jitter.iloc[0]) / 2.5,
                trace_collector=trace)
        new = pd.DataFrame(trace)
        print(vehicle, "max_diff", np.max(abs(new.x_after.to_numpy()-base.x_after.to_numpy())),
              "tail", new.x_after.iloc[-1], base.x_after.iloc[-1],
              "raw_tail", raw.FuelLevel.tail(5).to_list(), flush=True)
    model, metadata = load_fuel_state_classifier("models/fuel_state_classifier")
    for vehicle in ["21H-03221", "21H-02058"]:
        df = pd.read_csv(f"TienXuLy/{vehicle}_processed.csv")
        df = filter_with_ai_state(df, model=model, metadata=metadata, mode="realtime")
        trace = []
        out = filter_ai_enhanced_adaptive_realtime(df, config={"source_col": "FuelLevel", "trace_collector": trace})
        assert len(out) == len(df)
        assert np.isfinite(out).all()
        print(vehicle, "full replay", len(out), pd.DataFrame(trace).branch_selected.value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    main()
