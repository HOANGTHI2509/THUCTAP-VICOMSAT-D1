import os
import sys
sys.path.insert(0, os.path.abspath("."))
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import numpy as np
from src.core.filters.ai_enhanced_adaptive_realtime import (
    filter_ai_enhanced_adaptive_realtime,
    RealtimeAdaptiveKalmanState
)

def run_simulation(name, timestamps, raw_values, speeds=None, ai_states=None, capacity=300.0, jitter=0.8, event=6.0):
    n = len(raw_values)
    if speeds is None:
        speeds = [0.0] * n
    if ai_states is None:
        ai_states = ["STABLE_JITTER"] * n
        
    df = pd.DataFrame({
        "FuelTime": [t.isoformat() for t in timestamps],
        "FuelLevel": raw_values,
        "Speed": speeds,
        "AI_State": ai_states,
        "RollingStd": [jitter * 0.5] * n,
        "capacity_est": [capacity] * n,
        "flat_jitter_threshold": [jitter] * n,
        "event_threshold": [event] * n
    })
    
    trace = []
    clean = filter_ai_enhanced_adaptive_realtime(df, config={"trace_collector": trace})
    df_t = pd.DataFrame(trace)
    
    refuels = df_t[df_t["sustained_refuel"] == True]
    print(f"=== Scenario: {name} ===")
    print(f"  Total samples: {n}, Duration: {(timestamps[-1] - timestamps[0]).total_seconds()/60.0:.1f} min")
    print(f"  Refuels detected: {len(refuels)}")
    if len(refuels) > 0:
        for idx, r in refuels.iterrows():
            print(f"    -> Refuel at {r['fuel_time']}: raw_z={r['raw_z']}, x_after={r['x_after']}, branch={r['branch_selected']}")
    print(f"  Final x: {clean[-1]:.2f}, Final raw: {raw_values[-1]:.2f}, Diff: {clean[-1] - raw_values[-1]:.2f}")
    return df_t

def main():
    base_time = pd.Timestamp("2026-03-01 10:00:00")
    
    # 1. Dốc nhỏ kéo dài 120s cycle: +0.2 L/mẫu (0.10 L/min) kéo dài 60 phút (30 mẫu)
    t1 = [base_time + pd.Timedelta(minutes=2*i) for i in range(30)]
    # Anchor at 100. Jump to 108 then creep? Or pure creep from 100 to 120?
    # Test 1a: Pure creep from 100 to 115 (+0.5 L/sample = 0.25 L/min)
    r1a = [100.0 + 0.5 * i for i in range(30)]
    run_simulation("1a. Pure Creeping Ramp (+0.25 L/min, dt=2m)", t1, r1a)
    
    # Test 1b: Jump from 100 to 110, then creep +0.3 L/sample (+0.15 L/min)
    r1b = [100.0] * 5 + [110.0 + 0.3 * i for i in range(25)]
    run_simulation("1b. Jump +10L then Creep (+0.15 L/min, dt=2m)", t1, r1b)
    
    # Test 1c: Jump from 100 to 110, then oscillating ramp: 110 + 0.25*i + 1.2*sin(i)
    r1c = [100.0] * 5 + [110.0 + 0.25 * i + 1.2 * np.sin(i) for i in range(25)]
    run_simulation("1c. Jump +10L then Oscillating Ramp (+0.125 L/min, dt=2m)", t1, r1c)
    
    # Test 2a: 300s cycle (5 min), Pure creep from 100 to 120 (+1.0 L/sample = 0.20 L/min)
    t2 = [base_time + pd.Timedelta(minutes=5*i) for i in range(25)]
    r2a = [100.0 + 1.0 * i for i in range(25)]
    run_simulation("2a. 300s cycle: Pure Creep (+0.20 L/min, dt=5m)", t2, r2a)
    
    # Test 2b: 300s cycle, Real Refuel 100 -> 150 L, then flat
    r2b = [100.0] * 3 + [150.0] * 10
    t2b = [base_time + pd.Timedelta(minutes=5*i) for i in range(13)]
    run_simulation("2b. 300s cycle: Real Refuel (+50L flat, dt=5m)", t2b, r2b)
    
    # Test 2c: 300s cycle, Real Refuel with sloshing: 100 -> [148, 152, 149, 151, 150]
    r2c = [100.0] * 3 + [148.0, 152.0, 149.0, 151.0, 150.0, 150.0, 150.0]
    t2c = [base_time + pd.Timedelta(minutes=5*i) for i in range(10)]
    run_simulation("2c. 300s cycle: Real Refuel (+50L with jitter, dt=5m)", t2c, r2c)

if __name__ == "__main__":
    main()
