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
    filter_ai_enhanced_adaptive_realtime
)

def inspect_exact_812():
    model, metadata = load_fuel_state_classifier("models/fuel_state_classifier")
    df = pd.read_csv("TienXuLy/92H-02653_processed.csv")
    df_ai = filter_with_ai_state(df, model=model, metadata=metadata, mode="realtime")
    
    trace = []
    clean = filter_ai_enhanced_adaptive_realtime(df_ai, config={"trace_collector": trace})
    df_t = pd.DataFrame(trace)
    
    cols = [
        "idx", "fuel_time", "gap_minutes", "raw_z", "speed", "ai_state", "x_before", "x_after",
        "branch_selected", "update_mode", "sustained_refuel", "recent_refuel_steps",
        "flat_jitter", "event_threshold"
    ]
    sub = df_t.iloc[1798:1820]
    print(sub[cols].to_string())

inspect_exact_812()
