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
from src.core.filters.ai_state_filter import load_fuel_state_classifier, filter_with_ai_state
from src.core.filters.ai_enhanced_adaptive_realtime import (
    filter_ai_enhanced_adaptive_realtime,
    RealtimeAdaptiveKalmanState,
    _as_timestamp
)

def step_by_step_debug():
    model, metadata = load_fuel_state_classifier("models/fuel_state_classifier")
    df = pd.read_csv("TienXuLy/92H-02653_processed.csv")
    df_ai = filter_with_ai_state(df, model=model, metadata=metadata, mode="realtime")
    
    # We will run sample by sample with state
    state = RealtimeAdaptiveKalmanState()
    
    # Run up to row 1800
    for idx in range(1801):
        row_df = df_ai.iloc[[idx]]
        filter_ai_enhanced_adaptive_realtime(row_df, state=state)
        
    print("State at row 1800: x =", state.x, "anchor =", state.candidate_anchor)
    print("=" * 80)
    
    # Now step by step from 1801 to 1815
    for idx in range(1801, 1815):
        row_df = df_ai.iloc[[idx]]
        time_str = str(row_df["FuelTime"].values[0])
        raw_val = float(row_df["FuelLevel"].values[0])
        speed = float(row_df["Speed"].values[0])
        ai = str(row_df["AI_State"].values[0])
        
        # Capture state before step
        center_before = state.candidate_plateau_center
        samples_before = list(state.candidate_samples)
        times_before = list(state.candidate_sample_times)
        
        # Run 1 step
        out = filter_ai_enhanced_adaptive_realtime(row_df, state=state)
        
        # State after step
        print(f"Row {idx} | Time: {time_str} | Raw: {raw_val} | Speed: {speed} | AI: {ai}")
        print(f"  Center: {state.candidate_plateau_center} | Samples count: {len(state.candidate_samples)}")
        print(f"  Samples: {state.candidate_samples}")
        print(f"  Sample times: {state.candidate_sample_times}")
        
        # Compute window metrics
        win_k = min(len(state.candidate_samples), 5)
        win_samples = state.candidate_samples[-win_k:]
        win_times = state.candidate_sample_times[-win_k:]
        win_spread = max(win_samples) - min(win_samples) if win_samples else 0.0
        
        cur_t = _as_timestamp(time_str)
        w_dur = 0.0
        if cur_t is not None and win_times and win_times[0] is not None:
            w_start = _as_timestamp(win_times[0])
            if w_start is not None:
                w_dur = (cur_t - w_start).total_seconds() / 60.0
                
        print(f"  Win (k={win_k}): samples={win_samples} | times={win_times}")
        print(f"  Win spread: {win_spread} | Win duration: {w_dur} min")
        print(f"  Output x: {out[0]} | confirmed_level: {state.candidate_confirmed_level}")
        print("-" * 80)

step_by_step_debug()
